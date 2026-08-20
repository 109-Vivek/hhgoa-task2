# HH Goa 2026 Voice‑Enabled Indic RAG

This repository contains a **voice‑enabled Retrieval‑Augmented Generation (RAG)** system that satisfies the HH Goa 2026 Hackathon Task 2 requirements.

## Features

* **Speech‑to‑Text** – Sarvam AI STT (multi‑Indic, code‑mixed support)
* **Chunking** – Multi‑tiered, metadata‑aware chunking (atomic, sliding‑window, query‑anchor)
* **Hybrid Retrieval** – FAISS HNSW (dense) + BM25s (lexical) with Reciprocal Rank Fusion
* **LLM Generation** – Free or trial multi‑lingual models (Groq LLaMA‑3, Gemini, OpenAI)
* **Guardrails** – Input & output safety checks, abstention on low‑relevance
* **Latency** – Target <200 ms end‑to‑end, with P50/P70/P100 analytics
* **Deployment** – FastAPI + Streamlit demo, Docker‑ready

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-org/hhgoa.git
cd hhgoa

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your SARVAM_API_KEY, GROQ_API_KEY, etc.

# 5. Build indices (English, Hindi, Tamil)
python -m src.indexer --languages en hi ta

# 6. Run the benchmark harness
python -m src.benchmark --num-queries 100

# 7. Launch the demo
python -m src.app
```

The demo will expose a Streamlit UI where you can upload an audio file or record from the microphone.

## Project Structure

```
hhgoa/
├── task.md                 # Hackathon task description
├── specs.md                # Design decisions & architecture
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── stt/
│   │   ├── __init__.py
│   │   └── sarvam_stt.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── chunker.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── bge_embedder.py
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── dense_index.py
│   │   ├── bm25_index.py
│   │   └── hybrid_search.py
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── input_guard.py
│   │   └── output_guard.py
│   ├── harness/
│   │   └── orchestrator.py
│   ├── benchmark.py
│   ├── indexer.py
│   └── app.py
└── tests/
    └── test_pipeline.py
```

## License

MIT License – see `LICENSE`.
```