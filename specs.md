# Technical Specifications & System Architecture: Voice-Enabled Indic RAG

## 1. Project Overview & Objectives
This project is a high-performance, voice-enabled Retrieval-Augmented Generation (RAG) system built for the **HH Goa 2026 Hackathon (Task 2)**. The system enables users to speak queries in **Gujarati, Hindi, and Telugu** (`gu`, `hi`, `te`), accurately transcribes speech using Sarvam STT, performs hybrid sub-millisecond retrieval on the **MSMARCO-XI** dataset, and synthesizes grounded answers with strict guardrails and harness orchestration under a rigorous latency budget (<200ms pipeline target).

---

## 2. Core Architecture & Pipeline Flow

```
[ Voice Input (.wav/.mp3/mic) ]
               │
               ▼
   ┌───────────────────────┐
   │   Sarvam AI STT API   │  <-- Multi-Indic Speech-to-Text (gu, hi, te)
   └───────────┬───────────┘
               │ (Transcribed Query & Detected Lang)
               ▼
   ┌───────────────────────┐
   │ Input Guardrail Check │  <-- Safety / Off-topic / Lang validation
   └───────────┬───────────┘
               │
       ┌───────┴──────────────────┐
       ▼                          ▼
┌──────────────┐          ┌──────────────┐
│ Dense Embed  │          │ Lexical BM25 │
│ (BGE-M3 HNSW)│          │  (BM25s /    │
│  < 10-15ms   │          │   Tantivy)   │
└──────┬───────┘          └──────┬───────┘
       └──────────────┬───────────┘
                      ▼
        ┌───────────────────────────┐
        │ Hybrid Fusion & Reranking │ <-- RRF / Normalized Weighted Score
        └─────────────┬─────────────┘
                      ▼
        ┌───────────────────────────┐
        │ Harness & Grounding Check │ <-- Context Relevance Thresholding
        └─────────────┬─────────────┘
                      ▼
        ┌───────────────────────────┐
        │  Ultra-Fast LLM Generator │ <-- Groq / Gemini / OpenAI (JSON/Pydantic)
        └─────────────┬─────────────┘
                      ▼
        ┌───────────────────────────┐
        │ Output Guardrail & Metric │ <-- Hallucination check + P50/P70/P100 Logs
        └─────────────┬─────────────┘
                      ▼
              [ Grounded Answer ]
```

---

## 3. Key Design Decisions

### 3.1. Dataset: MSMARCO-XI
- **Source**: `ai4bharat/MSMARCO-XI` (Hugging Face).
- **Supported Languages**: Gujarati (`gu`), Hindi (`hi`), Telugu (`te`), and other Indic languages.
- **Direct Parquet Batch Loader**: Uses Hugging Face Hub downloads and PyArrow RecordBatch iteration to extract nested passages and queries robustly and incrementally.
- **Data Schema**:
  - `query_id`: Unique query identifier.
  - `passage_id`: Unique passage identifier.
  - `query`: Text in respective Indic language or English.
  - `passage`: Candidate text passage.
  - `label` / `relevance`: Binary or graded relevance flag.

### 3.2. Speech-to-Text (STT): Sarvam AI
- **Provider**: Sarvam AI (`saaras:v2` / Sarvam Speech-to-Text API).
- **Capabilities**:
  - Native multi-dialect Indic speech recognition (Hindi, Tamil, Indian English).
  - Code-mixed speech support (e.g., Hinglish, Tanglish).
  - Automatic language detection and timestamped transcriptions.
- **Fallback / Mock Mode**: Local Whisper or deterministic fallback harness for offline development and load testing.

### 3.3. Embedding Model: BAAI/bge-m3
- **Model**: `BAAI/bge-m3` (1024-dimensional dense representation).
- **Why BGE-M3?**:
  1. Native trilingual & multilingual support (100+ languages including Indic languages: Hindi, Tamil).
  2. Multi-granularity: Supports short queries up to 8192 token long contexts.
  3. Multi-functionality: Can output dense vectors, sparse lexical weights, and multi-vector ColBERT representations if needed.
- **Optimization**: ONNX runtime / TensorRT / FP16 quantized execution to minimize CPU/GPU embedding latency down to <15ms.


### 3.4. Multi-Tier Chunking Strategy (Vast & Metadata-Aware)
Rather than naive fixed-character chunking, we implement a multi-layered chunking taxonomy:

1. **Passage-Unit Chunking (Atomic Level)**:
   - Primary retrieval unit mapping 1:1 with MSMARCO-XI passage boundaries.
   - Preserves complete grammatical and semantic cohesion of the source passage.

2. **Hierarchical Sliding-Window with Overlap (Macro-to-Micro)**:
   - For extended documents: 256-token chunk size with 64-token overlap.
   - Sentence-boundary aware (uses Indic sentence tokenizers / NLTK).

3. **Query-Anchor & Metadata Augmentation**:
   - Chunks are enriched with metadata headers:
     `[LANG: {lang}] [DOC_ID: {id}] [TOPIC: {topic}] {passage_text}`
   - Improves cross-lingual dense alignment and filtered lexical lookup.

4. **Synthetic Question / Query-Passage Pairs**:
   - Leverage MSMARCO train queries to index dual query-passage anchors for bi-directional semantic matching.

### 3.5. Hybrid Retrieval & Search Indexing

#### Dense Semantic Index: FAISS with HNSW
- **Index Type**: `IndexHNSWFlat` / `IndexHNSWPQ` (M=32, efSearch=64, efConstruction=128).
- **Metric**: Inner Product (Cosine Similarity on L2-normalized vectors).
- **Search Latency**: ~2ms to 8ms for 100k+ vectors.

#### Lexical Keyword Index: Fast BM25 (BM25s / Tantivy)
- **Tokenization**: Multi-lingual Indic aware tokenization + English lowercase tokenizer.
- **Implementation**: Pure C/Rust-accelerated BM25 index (BM25s or Tantivy) operating in-memory or memory-mapped disk.
- **Search Latency**: ~1ms to 3ms.

#### Hybrid Fusion: Reciprocal Rank Fusion (RRF) & Weighted Score
- Combines Top-$K_{dense}$ and Top-$K_{lexical}$:
  $$\text{RRF\_Score}(d) = \frac{w_{dense}}{k + \text{rank}_{dense}(d)} + \frac{w_{lex}}{k + \text{rank}_{lex}(d)}$$
- Constant $k = 60$, dynamically adjusted weights per language ($w_{dense} = 0.65, w_{lex} = 0.35$).

### 3.6. Model Harness & Orchestration
The system uses a structured orchestration harness (via Pydantic/Instructor pattern):
1. **Schema Validation**: Guarantees strictly typed responses (`answer`, `sources`, `confidence`, `language`, `latency_breakdown`).
2. **Error Recovery & Retries**: Exponential backoff on STT/LLM API rate limits; fallback to dense-only retrieval if lexical index times out.
3. **Tool & Context Injector**: Formats retrieved context with passage ID citation tags `[1], [2]` and confidence metrics.

### 3.7. Guardrails & Abstention Architecture
1. **Input Guardrail (Pre-Retrieval)**:
   - Toxicity, prompt injection, and language validity filter.
   - Low-confidence STT transcriber filter (< 0.4 acoustic confidence prompts the user to repeat).
2. **Relevance & Abstention Guardrail (Post-Retrieval)**:
   - If maximum hybrid retrieval similarity $< \tau_{threshold}$ (e.g. 0.42), trigger abstention:
     *"I do not have enough relevant context in the database to answer this question accurately."*
3. **Grounding & Hallucination Guardrail (Post-Generation)**:
   - Verifies all generated claims against retrieved passage spans.
   - Prevents ungrounded hallucinated facts.

### 3.8. Latency Budget & Analytics Engine

#### 200ms Latency Budget Target:
| Stage | Component | Target Latency | Optimization Strategy |
|---|---|---|---|
| 1 | Speech-to-Text (STT) | 50ms - 80ms | Streaming audio chunking / Sarvam API |
| 2 | Guardrail (Input) | 2ms - 5ms | Fast regex / lightweight embedding filter |
| 3 | Query Embedding | 10ms - 15ms | ONNX / FP16 BGE-M3 model |
| 4 | Hybrid Retrieval | 5ms - 10ms | FAISS HNSW + BM25s parallel execution |
| 5 | LLM Generation | 60ms - 90ms | Ultra-fast inference (Groq LLaMA-3 / Sarvam) |
| 6 | Output Guardrail | 5ms - 10ms | Heuristic span grounding |
| **Total** | **End-to-End Pipeline** | **< 200ms** | Async pipeline + Concurrent async calls |

#### Latency Analytics:
- Real-time logging of timestamps per pipeline stage.
- Automatic calculation of **P50**, **P70**, and **P100 (Max)** latencies across benchmark query runs.
- Summary exported to JSON/CSV for hackathon submission evidence.

---

## 4. Quick Setup & Execution Guide

### 4.1. Prerequisites
- Python 3.10+ (macOS / Linux / Windows WSL)
- Git & Virtual Environment (`venv` or `conda`)
- API Keys:
  - `SARVAM_API_KEY`: Sarvam AI Speech/LLM API
  - `GROQ_API_KEY` or `OPENAI_API_KEY`: Fast generation LLM API

### 4.2. Installation Steps

```bash
# 1. Clone & create virtual environment
git clone <repo-url>
cd hhgoa
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your SARVAM_API_KEY, GROQ_API_KEY, etc.

# 4. Ingest and build indices for MSMARCO-XI (en, hi, ta)
python -m src.indexer --languages en hi ta --max-samples 50000

# 5. Run the Benchmark Harness (calculates P50, P70, P100)
python -m src.benchmark --num-queries 100

# 6. Launch Voice RAG Interactive Application (Streamlit / FastAPI)
python -m src.app
```

---

## 5. Verification & Validation Plan
- [x] High-performance multi-lingual embedding pipeline verified on Indic scripts (Devanagari, Tamil, Latin).
- [x] Sub-200ms latency budget mapped and parallelized across all stages.
- [x] Guardrails for hallucination and low-relevance queries configured with clear abstention policies.
- [x] Benchmark profiler ready to capture P50, P70, and P100 distribution across 100+ multi-lingual test queries.

