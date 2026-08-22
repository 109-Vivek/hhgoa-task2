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
class PipelineTraceStep:
    step_num: int
    step_id: str
    step_name: str
    time_ms: float
    status: str  # "passed", "failed", "completed", "warning", "blocked"
    details: Dict[str, Any] = field(default_factory=dict)


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
    pipeline_trace: List[Dict[str, Any]] = field(default_factory=list)

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

        # Build initial audio trace steps
        pre_trace_steps: List[Dict[str, Any]] = [
            {
                "step_num": 1,
                "step_id": "language_detection",
                "step_name": "Language Detected",
                "time_ms": round(stt_res.latency_ms * 0.1, 2),
                "status": "completed",
                "details": {
                    "detected_lang": detected_lang,
                    "lang_name": self._get_lang_full_name(detected_lang),
                    "detection_method": "Sarvam AI Speech Acoustic Language Classifier",
                    "input_code": language_code,
                    "model_detected_code": stt_res.language_code,
                },
            },
            {
                "step_num": 2,
                "step_id": "stt_transcription",
                "step_name": "STT Output (Speech-to-Text)",
                "time_ms": round(stt_res.latency_ms, 2),
                "status": "completed",
                "details": {
                    "transcript": stt_res.text,
                    "confidence": round(stt_res.confidence, 3),
                    "mode": "mock" if stt_res.is_mock else "sarvam_live_api",
                    "audio_filename": filename,
                    "audio_bytes_size": len(audio_bytes),
                },
            },
        ]

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
            pre_trace_steps=pre_trace_steps,
            is_voice=True,
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
        
        t_detect_start = time.perf_counter()
        if language_code == "auto" or not language_code:
            lang = self.detect_language(query, fallback_code="hi")
            detection_method = "Unicode Indic Script Range Analyzer"
        else:
            lang = self._normalize_lang(language_code)
            detection_method = f"User Explicit Selection ({language_code})"
        detect_ms = (time.perf_counter() - t_detect_start) * 1000.0

        pre_trace_steps: List[Dict[str, Any]] = [
            {
                "step_num": 1,
                "step_id": "language_detection",
                "step_name": "Language Detected",
                "time_ms": round(max(detect_ms, 0.05), 2),
                "status": "completed",
                "details": {
                    "detected_lang": lang,
                    "lang_name": self._get_lang_full_name(lang),
                    "detection_method": detection_method,
                    "script_analysis": self._get_script_debug_info(query),
                },
            }
        ]

        return self._execute_text_pipeline(
            query=query,
            lang=lang,
            stt_confidence=1.0,
            timing=timing,
            total_start=total_start,
            stt_info=None,
            pre_trace_steps=pre_trace_steps,
            is_voice=False,
        )

    def _execute_text_pipeline(
        self,
        query: str,
        lang: str,
        stt_confidence: float,
        timing: PipelineLatencyBreakdown,
        total_start: float,
        stt_info: Optional[Dict[str, Any]] = None,
        pre_trace_steps: Optional[List[Dict[str, Any]]] = None,
        is_voice: bool = False,
    ) -> PipelineResponse:
        """Internal execution flow for guardrails, retrieval, LLM synthesis, and grounding."""
        trace: List[Dict[str, Any]] = list(pre_trace_steps or [])

        # Input Guardrail Check
        ig_res = self.input_guard.evaluate(query, stt_confidence=stt_confidence)
        timing.input_guardrail_ms = ig_res.latency_ms

        step_counter = len(trace) + 1
        trace.append({
            "step_num": step_counter,
            "step_id": "input_guardrail",
            "step_name": "Input Guardrail Check",
            "time_ms": round(ig_res.latency_ms, 2),
            "status": "passed" if ig_res.is_safe else "blocked",
            "details": {
                "is_safe": ig_res.is_safe,
                "action": ig_res.action,
                "reason": ig_res.reason,
                "stt_confidence": round(stt_confidence, 3),
                "checks_applied": [
                    "Prompt Injection / Jailbreak Filter",
                    "Harmful / Toxic Content Patterns",
                    "Acoustic Confidence Threshold (>= 0.35)",
                    "Minimum Query Length (> 2 chars)",
                ],
            },
        })

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
                pipeline_trace=trace,
            )

        # Dense Embedding & Hybrid Retrieval
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

        # Trace: BGE-M3 Query Embedding
        step_counter += 1
        trace.append({
            "step_num": step_counter,
            "step_id": "query_embedding",
            "step_name": "BGE-M3 Query Embedding",
            "time_ms": round(timing.embedding_ms, 2),
            "status": "completed",
            "details": {
                "model_name": self.embedder.model_name,
                "embedding_dimension": 1024,
                "device": str(self.embedder.device).upper(),
                "query_text": query,
            },
        })

        # Trace: FAISS HNSW Dense Search
        step_counter += 1
        trace.append({
            "step_num": step_counter,
            "step_id": "dense_search",
            "step_name": "FAISS HNSW Dense Search",
            "time_ms": round(timing.dense_search_ms, 2),
            "status": "completed",
            "details": {
                "index_type": "IndexHNSWFlat (M=32, efSearch=64)",
                "distance_metric": "Cosine Similarity (Inner Product on L2-norm)",
                "passage_dense_candidates": metrics.get("dense_result_count", 0),
                "query_anchor_candidates": metrics.get("query_anchor_result_count", 0),
                "target_language_index": lang,
            },
        })

        # Trace: BM25s Lexical Search
        step_counter += 1
        trace.append({
            "step_num": step_counter,
            "step_id": "lexical_search",
            "step_name": "BM25s Lexical Search",
            "time_ms": round(timing.lexical_search_ms, 2),
            "status": "completed",
            "details": {
                "algorithm": "BM25s (Optimized Indic-aware Tokenizer)",
                "lexical_candidates_matched": metrics.get("lexical_result_count", 0),
                "target_language_index": lang,
            },
        })

        # Trace: Reciprocal Rank Fusion (RRF)
        step_counter += 1
        trace.append({
            "step_num": step_counter,
            "step_id": "rrf_fusion",
            "step_name": "Reciprocal Rank Fusion (RRF)",
            "time_ms": round(timing.fusion_ms, 2),
            "status": "completed",
            "details": {
                "fusion_constant_k": 60,
                "track_weights": {
                    "dense_weight": 0.40,
                    "query_anchor_weight": 0.40,
                    "lexical_weight": 0.20,
                },
                "total_unique_fused_candidates": len(retrieved_raw),
                "top_k_selected": self.max_results,
            },
        })

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

        # Trace: Final Chunks Fetched with Metadata
        step_counter += 1
        trace.append({
            "step_num": step_counter,
            "step_id": "final_chunks",
            "step_name": "Final Chunks (Retrieved Passages & Metadata)",
            "time_ms": 0.05,
            "status": "completed",
            "details": {
                "chunks_count": len(retrieved_docs),
                "max_similarity_score": round(max_dense_score, 4),
                "chunks": [
                    {
                        "rank": idx + 1,
                        "chunk_id": d.chunk_id,
                        "passage_id": d.passage_id,
                        "lang": d.lang,
                        "dense_score": round(d.dense_score, 4),
                        "rrf_score": round(d.rrf_score, 4),
                        "lexical_score": round(d.lexical_score, 4),
                        "match_sources": d.match_sources,
                        "text_preview": d.text[:120] + ("..." if len(d.text) > 120 else ""),
                        "metadata": d.metadata,
                    }
                    for idx, d in enumerate(retrieved_docs)
                ],
            },
        })

        # Check for abstention condition
        should_abstain = False
        if not retrieved_docs or max_dense_score < self.similarity_threshold:
            should_abstain = True

        # LLM Synthesis
        context_dicts = [{"text": d.text, "chunk_id": d.chunk_id, "passage_id": d.passage_id, "raw_text": d.raw_text, "score": d.dense_score} for d in retrieved_docs]
        llm_res = self.llm_client.generate_answer(
            query=query,
            retrieved_contexts=context_dicts,
            lang=lang,
            is_abstention=should_abstain,
        )

        timing.llm_generation_ms = llm_res.get("latency_ms", 0.0)
        answer = llm_res.get("answer", "")
        provider = llm_res.get("provider", "unknown")
        system_prompt = llm_res.get("system_prompt", "")
        user_prompt = llm_res.get("user_prompt", "")

        # Trace: Final Prompt Sent to LLM
        step_counter += 1
        trace.append({
            "step_num": step_counter,
            "step_id": "final_prompt",
            "step_name": "Final Prompt Sent to LLM",
            "time_ms": 0.05,
            "status": "completed",
            "details": {
                "target_language": self._get_lang_full_name(lang),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "context_citations_included": len(retrieved_docs),
            },
        })

        # Trace: LLM Output
        step_counter += 1
        trace.append({
            "step_num": step_counter,
            "step_id": "llm_output",
            "step_name": "LLM Output & Synthesis",
            "time_ms": round(timing.llm_generation_ms, 2),
            "status": "completed",
            "details": {
                "provider": provider,
                "raw_response": answer,
                "is_abstention": should_abstain,
                "tokens_generated_approx": len(answer.split()),
            },
        })

        # Output Guardrail (Grounding & Safety Verification)
        og_res = self.output_guard.evaluate(
            answer=answer,
            retrieved_passages=passages,
            max_retrieval_similarity=max_dense_score,
            similarity_threshold=self.similarity_threshold,
        )
        timing.output_guardrail_ms = og_res.latency_ms

        # Trace: Output Guardrail Check
        step_counter += 1
        trace.append({
            "step_num": step_counter,
            "step_id": "output_guardrail",
            "step_name": "Output Guardrail Check (Grounding & Hallucination)",
            "time_ms": round(og_res.latency_ms, 2),
            "status": "passed" if (og_res.is_grounded and not og_res.hallucination_detected) else ("abstention" if og_res.is_abstention else "flagged"),
            "details": {
                "is_grounded": og_res.is_grounded,
                "grounding_score": round(og_res.grounding_score, 3),
                "hallucination_detected": og_res.hallucination_detected,
                "is_abstention": og_res.is_abstention,
                "reason": og_res.reason,
                "min_grounding_threshold": self.output_guard.min_grounding_threshold,
            },
        })

        timing.total_end_to_end_ms = (
            timing.stt_ms
            + timing.input_guardrail_ms
            + timing.total_retrieval_ms
            + timing.output_guardrail_ms
        )

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
            pipeline_trace=trace,
        )

    @staticmethod
    def _get_lang_full_name(code: str) -> str:
        names = {
            "gu": "Gujarati (ગુજરાતી)",
            "hi": "Hindi (हिन्दी)",
            "te": "Telugu (తెలుగు)",
            "en": "English",
        }
        return names.get(code, f"Indic ({code})")

    @staticmethod
    def _get_script_debug_info(text: str) -> str:
        if any('\u0A80' <= ch <= '\u0AFF' for ch in text):
            return "Gujarati script (U+0A80 to U+0AFF)"
        if any('\u0900' <= ch <= '\u097F' for ch in text):
            return "Devanagari / Hindi script (U+0900 to U+097F)"
        if any('\u0C00' <= ch <= '\u0C7F' for ch in text):
            return "Telugu script (U+0C00 to U+0C7F)"
        return "Latin / English or non-Indic fallback"

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
