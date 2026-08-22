# HH Goa 2026 Voice‑Enabled Indic RAG

This repository contains a **voice‑enabled Retrieval‑Augmented Generation (RAG)** system that satisfies the HH Goa 2026 Hackathon Task 2 requirements.

## Features

* **Speech‑to‑Text** – Sarvam AI STT (multi‑Indic, code‑mixed support)
* **Chunking** – Multi‑tiered, metadata‑aware chunking (atomic, sliding‑window, query‑anchor)
* **Hybrid Retrieval** – FAISS HNSW (dense) + BM25s (lexical) with Reciprocal Rank Fusion
* **LLM Generation** – Free or trial multi‑lingual models (Groq LLaMA‑3, Gemini, OpenAI)
* **Guardrails** – Input & output safety checks, abstention on low‑relevance
* **Latency** – Target <200 ms end‑to‑end, with P50/P70/P100 analytics
* **Deployment** – FastAPI + React frontend, Docker‑ready

Try it out : [https://hhgoasge.duckdns.org](https://hhgoasge.duckdns.org/) 

# [Demo Video Link](https://x.com/vivekyadavnitt/status/2091240572967436349)
