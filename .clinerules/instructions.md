# Indic Voice-RAG Project Context & Guidelines

## 1. Project Overview
- **Task:** HH Goa 2026 Hackathon Task 2 (Voice-Enabled Indic RAG).
- **Core Pipeline:** Voice / Audio Input → Sarvam AI STT (`saaras:v2`) → Input Guardrail Filter → Multi-Tier Chunking / Hybrid Retrieval (FAISS HNSW + BM25s with RRF) → Resilient LLM Harness & Grounding Check → Output Guardrail (Abstention & Hallucination verification).
- **Languages Supported:** English (`en`), Hindi (`hi`), Tamil (`ta`) with automatic script and audio dialect detection.
- **Latency Budget:** Target < 200 ms end-to-end pipeline latency. Realized P50: ~48-65 ms.

## 2. Architecture & Tech Stack
- **Dense Embedding Model:** `BAAI/bge-m3` (1024-dim, L2 normalized).
- **Vector Search:** `faiss-cpu` / `faiss.IndexHNSWFlat` (Inner Product, M=32, efSearch=64).
- **Lexical Search:** `bm25s` with Indic Devanagari & Tamil tokenization.
- **Fusion:** Reciprocal Rank Fusion (RRF, $k=60$).
- **Harness & LLM:** `ResilientLLMClient` supporting Groq LLaMA-3.3, Gemini, OpenAI, xAI, and fast local Indic extractive grounding fallback (<2ms).
- **Backend API:** FastAPI (`src/server.py`) on port 8000 with CORS and REST endpoints.
- **Frontend UI:** React + Vite + TypeScript (`frontend/`) on port 5173 with Web Audio API recording, live waveform visualizer, waterfall latency breakdown, benchmark explorer, and index manager.

## 3. Quick Run Commands
```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Build indices
python -m src.indexer --languages en hi ta --use-sample

# 3. Run automated tests
python -m unittest tests/test_pipeline.py

# 4. Run latency benchmark harness (P50/P70/P100)
python -m src.benchmark --num-queries 15

# 5. Start Backend API
python -m src.server

# 6. Start React Frontend
cd frontend && npm run dev
```