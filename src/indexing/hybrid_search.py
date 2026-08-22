import time
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from src.indexing.dense_index import FAISSIndex
from src.indexing.bm25_index import BM25Index
from src.embeddings.bge_embedder import BGEEmbedder, get_embedder
from src.config import (
    RRF_K,
    DENSE_WEIGHT,
    LEXICAL_WEIGHT,
    QUERY_ANCHOR_WEIGHT,
    SIMILARITY_THRESHOLD,
    MAX_RETRIEVAL_RESULTS,
)


class HybridSearchEngine:
    """
    Sub-millisecond Dual-Track Hybrid Retrieval Engine combining:
    1. Query-Anchor Dense Index (Query-to-Query Intent Matching)
    2. Passage Dense Index (Direct Query-to-Passage Semantic Matching)
    3. Lexical BM25s Index (Keyword / Exact Entity Matching)
    Executed in parallel via thread workers and fused using Tri-Track Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_index: FAISSIndex,
        lexical_index: BM25Index,
        query_dense_index: Optional[FAISSIndex] = None,
        embedder: Optional[BGEEmbedder] = None,
        rrf_k: int = RRF_K,
        dense_weight: float = DENSE_WEIGHT,
        lexical_weight: float = LEXICAL_WEIGHT,
        query_anchor_weight: float = QUERY_ANCHOR_WEIGHT,
    ):
        self.dense_index = dense_index
        self.lexical_index = lexical_index
        self.query_dense_index = query_dense_index
        self.embedder = embedder or get_embedder()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.query_anchor_weight = query_anchor_weight

    def search(
        self,
        query: str,
        top_k: int = MAX_RETRIEVAL_RESULTS,
        lang: str = "en",
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> Tuple[List[Dict[str, Any]], float, Dict[str, float]]:
        """
        Executes parallel query-anchor + passage-dense + lexical search and tri-track RRF fusion.
        Returns: (fused_results, retrieval_time_ms, detailed_metrics)
        """
        start_time = time.perf_counter()
        timing = {}

        # 1. Query Embedding (computed once for both dense indices)
        embed_start = time.perf_counter()
        query_vec = self.embedder.encode_query(query)
        timing["embedding_ms"] = (time.perf_counter() - embed_start) * 1000.0

        # 2. Parallel Search Execution with Granular Timings
        search_start = time.perf_counter()
        
        def timed_dense():
            t0 = time.perf_counter()
            res = self.dense_index.search(query_vec, top_k * 2)
            return res, (time.perf_counter() - t0) * 1000.0

        def timed_query():
            if self.query_dense_index and self.query_dense_index.count() > 0:
                t0 = time.perf_counter()
                res = self.query_dense_index.search(query_vec, top_k * 2)
                return res, (time.perf_counter() - t0) * 1000.0
            return [], 0.0

        def timed_lexical():
            t0 = time.perf_counter()
            res = self.lexical_index.search(query, top_k * 2)
            return res, (time.perf_counter() - t0) * 1000.0

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_passage = executor.submit(timed_dense)
            future_query = executor.submit(timed_query)
            future_lexical = executor.submit(timed_lexical)

            passage_dense_results, dense_time_ms = future_passage.result()
            query_anchor_results, qa_time_ms = future_query.result()
            lexical_results, lexical_time_ms = future_lexical.result()

        timing["dense_search_ms"] = dense_time_ms
        timing["query_anchor_search_ms"] = qa_time_ms
        timing["lexical_search_ms"] = lexical_time_ms
        timing["parallel_search_ms"] = (time.perf_counter() - search_start) * 1000.0
        timing["dense_result_count"] = len(passage_dense_results)
        timing["query_anchor_result_count"] = len(query_anchor_results)
        timing["lexical_result_count"] = len(lexical_results)

        # 3. Tri-Track Reciprocal Rank Fusion (RRF) & Passage Deduplication
        fusion_start = time.perf_counter()
        doc_map: Dict[str, Dict[str, Any]] = {}
        scores_map: Dict[str, float] = {}
        dense_sims: Dict[str, float] = {}
        q2q_sims: Dict[str, float] = {}
        lexical_scores: Dict[str, float] = {}
        match_sources: Dict[str, List[str]] = {}

        def _get_doc_id(meta: Dict[str, Any]) -> str:
            return str(meta.get("passage_id") or meta.get("doc_id") or meta.get("chunk_id", ""))

        # Track A: Query-to-Query Anchor Matches (Highest Intent Precision)
        # Look up matched queries, then pull ALL associated passage chunks from the passage dense index
        for rank, (meta, score) in enumerate(query_anchor_results, start=1):
            doc_id = _get_doc_id(meta)
            if not doc_id:
                continue

            # Find all matching passage chunks in the passage dense index
            matching_chunks = [
                chunk_meta for chunk_meta in self.dense_index.metadata_store
                if _get_doc_id(chunk_meta) == doc_id or str(chunk_meta.get("passage_id", "")) == doc_id
            ]

            if matching_chunks:
                for chunk_meta in matching_chunks:
                    c_id = _get_doc_id(chunk_meta)
                    if not c_id:
                        continue
                    if c_id not in doc_map or not doc_map[c_id].get("raw_text"):
                        doc_map[c_id] = chunk_meta
                    q2q_sims[c_id] = max(q2q_sims.get(c_id, 0.0), float(score))
                    rrf_score = self.query_anchor_weight * (1.0 / (self.rrf_k + rank))
                    scores_map[c_id] = scores_map.get(c_id, 0.0) + rrf_score
                    match_sources.setdefault(c_id, []).append(f"query_anchor (sim={score:.3f})")
            else:
                if doc_id not in doc_map:
                    doc_map[doc_id] = {
                        "chunk_id": meta.get("anchor_id", f"{doc_id}_qa"),
                        "passage_id": doc_id,
                        "text": meta.get("passage_text", meta.get("text", "")),
                        "raw_text": meta.get("passage_text", meta.get("raw_text", meta.get("text", ""))),
                        "lang": meta.get("lang", lang),
                        "metadata": meta.get("metadata", {}),
                    }
                q2q_sims[doc_id] = max(q2q_sims.get(doc_id, 0.0), float(score))
                rrf_score = self.query_anchor_weight * (1.0 / (self.rrf_k + rank))
                scores_map[doc_id] = scores_map.get(doc_id, 0.0) + rrf_score
                match_sources.setdefault(doc_id, []).append(f"query_anchor (sim={score:.3f})")

        # Track B: Direct Query-to-Passage Dense Matches (Semantic Recall)
        for rank, (meta, score) in enumerate(passage_dense_results, start=1):
            doc_id = _get_doc_id(meta)
            if not doc_id:
                continue
            if doc_id not in doc_map or not doc_map[doc_id].get("raw_text"):
                doc_map[doc_id] = meta
            dense_sims[doc_id] = max(dense_sims.get(doc_id, 0.0), float(score))
            rrf_score = self.dense_weight * (1.0 / (self.rrf_k + rank))
            scores_map[doc_id] = scores_map.get(doc_id, 0.0) + rrf_score
            match_sources.setdefault(doc_id, []).append(f"passage_dense (sim={score:.3f})")

        # Track C: Lexical BM25 Matches (Keyword / Entity Precision)
        for rank, (meta, score) in enumerate(lexical_results, start=1):
            doc_id = _get_doc_id(meta)
            if not doc_id:
                continue
            if doc_id not in doc_map or not doc_map[doc_id].get("raw_text"):
                doc_map[doc_id] = meta
            lexical_scores[doc_id] = float(score)
            rrf_score = self.lexical_weight * (1.0 / (self.rrf_k + rank))
            scores_map[doc_id] = scores_map.get(doc_id, 0.0) + rrf_score
            match_sources.setdefault(doc_id, []).append(f"lexical_bm25 (score={score:.2f})")

        timing["fusion_ms"] = (time.perf_counter() - fusion_start) * 1000.0

        # Sort documents by fused RRF score
        sorted_docs = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            meta = doc_map.get(doc_id, {})
            q_score = q2q_sims.get(doc_id, 0.0)
            p_score = dense_sims.get(doc_id, 0.0)
            effective_dense = max(q_score, p_score)
            
            item = {
                **meta,
                "rrf_score": rrf_score,
                "dense_score": effective_dense,
                "q2q_score": q_score,
                "passage_dense_score": p_score,
                "lexical_score": lexical_scores.get(doc_id, 0.0),
                "match_sources": match_sources.get(doc_id, []),
            }
            final_results.append(item)

        total_retrieval_ms = (time.perf_counter() - start_time) * 1000.0
        timing["total_retrieval_ms"] = total_retrieval_ms

        return final_results, total_retrieval_ms, timing

