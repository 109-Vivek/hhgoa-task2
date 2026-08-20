import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from src.config import (
    INDEX_DIR,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANG,
    SIMILARITY_THRESHOLD,
    MAX_RETRIEVAL_RESULTS,
)
from src.stt.sarvam_stt import SarvamSTT, TranscriptionResult
from src.embeddings.bge_embedder import get_embedder, BGEEmbedder
from src.indexing.dense_index import FAISSIndex
from src.indexing.bm25_index import BM25Index
from src.indexing.hybrid_search import HybridSearchEngine
from src.guardrails.input_guard import InputGuardrail, GuardrailCheckResult
from src.guardrails.output_guard import OutputGuardrail, GroundingCheckResult
from src.harness.llm_client import ResilientLLMClient


@dataclass
class RetrievedDocument:
    chunk_id: str
    passage_id: str
    text: str
    raw_text: str
    lang: str
    rrf_score: float
    dense_score: float
    q2q_score: float = 0.0
    passage_dense_score: float = 0.0
    lexical_score: float = 0.0
    match_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineLatencyBreakdown:
    stt_ms: float = 0.0
    input_guardrail_ms: float = 0.0
    embedding_ms: float = 0.0
    dense_search_ms: float = 0.0
    lexical_search_ms: float = 0.0
    fusion_ms: float = 0.0
    total_retrieval_ms: float = 0.0
    llm_generation_ms: float = 0.0
    output_guardrail_ms: float = 0.0
    total_end_to_end_ms: float = 0.0


@dataclass
class PipelineResponse:
    query: str
    detected_lang: str
    answer: str
    retrieved_documents: List[RetrievedDocument]
    input_guard: Dict[str, Any]
    output_guard: Dict[str, Any]
    latency: PipelineLatencyBreakdown
    is_abstention: bool
    provider: str
    stt_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VoiceRAGOrchestrator:
    """
    Main orchestration harness for the Voice-Enabled Indic RAG system.
    Coordinates STT, Input Guardrails, Multilingual Dual-Track Hybrid Search (Query-Anchor + Passage FAISS + BM25s),
    Context Grounding & Abstention, LLM Synthesis, and Output Guardrails.
    """

    def __init__(
        self,
        index_dir: Path = INDEX_DIR,
        embedder: Optional[BGEEmbedder] = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        max_results: int = MAX_RETRIEVAL_RESULTS,
    ):
        self.index_dir = Path(index_dir)
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results

        # Core Components
        self.stt = SarvamSTT()
        self.input_guard = InputGuardrail()
        self.output_guard = OutputGuardrail()
        self.llm_client = ResilientLLMClient()
        self.embedder = embedder or get_embedder()

        # Multilingual Hybrid Search Engines
        self.search_engines: Dict[str, HybridSearchEngine] = {}
        self._load_indices()

    def _load_indices(self):
        """Loads passage dense, query anchor dense, and lexical indices for each supported language."""
        for lang in SUPPORTED_LANGUAGES:
            lang_dir = self.index_dir / lang
            dense = FAISSIndex()
            query_dense = FAISSIndex()
            bm25 = BM25Index()

            dense_loaded = dense.load(lang_dir, index_name="faiss.index", meta_name="faiss_meta.json")
            query_loaded = query_dense.load(lang_dir, index_name="query_faiss.index", meta_name="query_faiss_meta.json")
            bm25_loaded = bm25.load(lang_dir)

            if dense_loaded and bm25_loaded:
                q_count = query_dense.count() if query_loaded else 0
                print(f"[Orchestrator] Loaded indices for language '{lang}' ({dense.count()} passage docs, {q_count} query anchors)")
            else:
                print(f"[Orchestrator] Notice: Index for language '{lang}' not yet built at {lang_dir}")

            self.search_engines[lang] = HybridSearchEngine(
                dense_index=dense,
                lexical_index=bm25,
                query_dense_index=query_dense if query_loaded else None,
                embedder=self.embedder,
            )

    def process_audio(
        self,
        audio_bytes: bytes,
        filename: str = "input.wav",
        language_code: str = "hi-IN",
    ) -> PipelineResponse:
        """
        End-to-end processing pipeline starting from raw voice audio.
        Voice -> STT -> Input Guard -> Hybrid Retrieval -> LLM -> Output Guard.
        """
        total_start = time.perf_counter()
        timing = PipelineLatencyBreakdown()

        # 1. Speech-to-Text
        stt_res = self.stt.transcribe_audio_bytes(
            audio_bytes, filename=filename, language_code=language_code
        )
        timing.stt_ms = stt_res.latency_ms

        query = stt_res.text
        # Map audio language code (e.g., 'hi-IN' -> 'hi')
        detected_lang = self._normalize_lang(stt_res.language_code)

        # Proceed with text processing
        stt_info = {
            "text": stt_res.text,
            "is_mock": stt_res.is_mock,
            "confidence": stt_res.confidence,
            "latency_ms": stt_res.latency_ms,
            "mode": "mock" if stt_res.is_mock else "sarvam_live",
        }
        return self._execute_text_pipeline(
            query=query,
            lang=detected_lang,
            stt_confidence=stt_res.confidence,
            timing=timing,
            total_start=total_start,
            stt_info=stt_info,
        )

    def process_query(
        self,
        query: str,
        language_code: str = "auto",
    ) -> PipelineResponse:
        """
        Direct text-query execution pipeline (bypassing STT).
        """
        total_start = time.perf_counter()
        timing = PipelineLatencyBreakdown()
        
        if language_code == "auto" or not language_code:
            lang = self.detect_language(query, fallback_code="en")
        else:
            lang = self._normalize_lang(language_code)

        return self._execute_text_pipeline(
            query=query,
            lang=lang,
            stt_confidence=1.0,
            timing=timing,
            total_start=total_start,
            stt_info=None,
        )

    def _execute_text_pipeline(
        self,
        query: str,
        lang: str,
        stt_confidence: float,
        timing: PipelineLatencyBreakdown,
        total_start: float,
        stt_info: Optional[Dict[str, Any]] = None,
    ) -> PipelineResponse:
        """Internal execution flow for guardrails, retrieval, LLM synthesis, and grounding."""

        # 2. Input Guardrail
        ig_res = self.input_guard.evaluate(query, stt_confidence=stt_confidence)
        timing.input_guardrail_ms = ig_res.latency_ms

        if not ig_res.is_safe:
            print(f"[Orchestrator] Input guardrail triggered: {ig_res.reason}")
            timing.total_end_to_end_ms = (time.perf_counter() - total_start) * 1000.0
            return PipelineResponse(
                query=query,
                detected_lang=lang,
                answer=f"Request blocked: {ig_res.reason}",
                retrieved_documents=[],
                input_guard=asdict(ig_res),
                output_guard=asdict(GroundingCheckResult(
                    is_grounded=True,
                    grounding_score=1.0,
                    is_abstention=True,
                    hallucination_detected=False,
                    reason="Input blocked by guardrail.",
                    latency_ms=0.0,
                )),
                latency=timing,
                is_abstention=True,
                provider="input_guardrail",
                stt_info=stt_info,
            )

        # 3. Dense Embedding & Hybrid Retrieval
        engine = self.search_engines.get(lang)
        if not engine:
            engine = self.search_engines.get(DEFAULT_LANG)

        retrieved_raw, total_retrieval_ms, metrics = engine.search(
            query=query,
            top_k=self.max_results,
            lang=lang,
            similarity_threshold=self.similarity_threshold,
        )

        timing.embedding_ms = metrics.get("embedding_ms", 0.0)
        timing.dense_search_ms = metrics.get("dense_search_ms", 0.0)
        timing.lexical_search_ms = metrics.get("lexical_search_ms", 0.0)
        timing.fusion_ms = metrics.get("fusion_ms", 0.0)
        timing.total_retrieval_ms = total_retrieval_ms

        retrieved_docs: List[RetrievedDocument] = []
        passages: List[str] = []
        max_dense_score = 0.0

        for item in retrieved_raw:
            d_score = float(item.get("dense_score", 0.0))
            if d_score > max_dense_score:
                max_dense_score = d_score
            doc = RetrievedDocument(
                chunk_id=str(item.get("chunk_id", "")),
                passage_id=str(item.get("passage_id", item.get("doc_id", ""))),
                text=str(item.get("text", "")),
                raw_text=str(item.get("raw_text", item.get("text", ""))),
                lang=str(item.get("lang", lang)),
                rrf_score=float(item.get("rrf_score", 0.0)),
                dense_score=d_score,
                q2q_score=float(item.get("q2q_score", 0.0)),
                passage_dense_score=float(item.get("passage_dense_score", 0.0)),
                lexical_score=float(item.get("lexical_score", 0.0)),
                match_sources=item.get("match_sources", []),
                metadata=item.get("metadata", {}),
            )
            retrieved_docs.append(doc)
            passages.append(doc.text)

        # Check for abstention condition
        should_abstain = False
        if not retrieved_docs or max_dense_score < self.similarity_threshold:
            should_abstain = True

        # 4. LLM Synthesis
        context_dicts = [{"text": d.text, "chunk_id": d.chunk_id, "score": d.dense_score} for d in retrieved_docs]
        llm_res = self.llm_client.generate_answer(
            query=query,
            retrieved_contexts=context_dicts,
            lang=lang,
            is_abstention=should_abstain,
        )

        timing.llm_generation_ms = llm_res.get("latency_ms", 0.0)
        answer = llm_res.get("answer", "")
        provider = llm_res.get("provider", "unknown")

        # 5. Output Guardrail (Grounding & Safety Verification)
        og_res = self.output_guard.evaluate(
            answer=answer,
            retrieved_passages=passages,
            max_retrieval_similarity=max_dense_score,
            similarity_threshold=self.similarity_threshold,
        )
        timing.output_guardrail_ms = og_res.latency_ms

        timing.total_end_to_end_ms = (time.perf_counter() - total_start) * 1000.0

        return PipelineResponse(
            query=query,
            detected_lang=lang,
            answer=answer,
            retrieved_documents=retrieved_docs,
            input_guard=asdict(ig_res),
            output_guard=asdict(og_res),
            latency=timing,
            is_abstention=og_res.is_abstention or should_abstain,
            provider=provider,
            stt_info=stt_info,
        )

    @staticmethod
    def _normalize_lang(lang_code: str) -> str:
        if not lang_code:
            return DEFAULT_LANG
        code = lang_code.lower()
        if "gu" in code:
            return "gu"
        if "hi" in code:
            return "hi"
        if "te" in code:
            return "te"
        return DEFAULT_LANG

    @staticmethod
    def detect_language(text: str, fallback_code: str = DEFAULT_LANG) -> str:
        if not text:
            return VoiceRAGOrchestrator._normalize_lang(fallback_code)
        # Check Gujarati range (\u0A80-\u0AFF)
        if any('\u0A80' <= ch <= '\u0AFF' for ch in text):
            return "gu"
        # Check Devanagari range (Hindi) (\u0900-\u097F)
        if any('\u0900' <= ch <= '\u097F' for ch in text):
            return "hi"
        # Check Telugu range (\u0C00-\u0C7F)
        if any('\u0C00' <= ch <= '\u0C7F' for ch in text):
            return "te"
        return VoiceRAGOrchestrator._normalize_lang(fallback_code)
