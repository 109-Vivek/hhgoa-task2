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

        # 2. Parallel Search Execution
        search_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_passage = executor.submit(self.dense_index.search, query_vec, top_k * 2)
            
            future_query = None
            if self.query_dense_index and self.query_dense_index.count() > 0:
                future_query = executor.submit(self.query_dense_index.search, query_vec, top_k * 2)

            future_lexical = executor.submit(self.lexical_index.search, query, top_k * 2)

            passage_dense_results = future_passage.result()
            query_anchor_results = future_query.result() if future_query else []
            lexical_results = future_lexical.result()

        timing["parallel_search_ms"] = (time.perf_counter() - search_start) * 1000.0

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

        # Track A: Query-to-Query Anchor Matches → Passage Lookup via Metadata
        # Step 1: Collect matched passage_ids from query anchor results
        anchor_passage_ids: Dict[str, float] = {}  # passage_id → best similarity score
        for rank, (meta, score) in enumerate(query_anchor_results, start=1):
            pid = str(meta.get("passage_id") or "")
            if pid:
                # Keep best similarity score per passage_id; also accumulate RRF contribution
                if pid not in anchor_passage_ids or score > anchor_passage_ids[pid]:
                    anchor_passage_ids[pid] = float(score)

        # Step 2: Look up ALL passage chunks in the passage dense index that belong to matched passage_ids
        # Build a passage_id → list of chunk metadata map from the passage index store
        passage_chunk_map: Dict[str, List[Dict[str, Any]]] = {}
        for chunk_meta in self.dense_index.metadata_store:
            pid = str(chunk_meta.get("passage_id") or "")
            if pid in anchor_passage_ids:
                passage_chunk_map.setdefault(pid, []).append(chunk_meta)

        # Step 3: Register all retrieved chunks into the fusion maps with RRF scores
        for rank, (pid, sim_score) in enumerate(
            sorted(anchor_passage_ids.items(), key=lambda x: x[1], reverse=True), start=1
        ):
            chunks = passage_chunk_map.get(pid, [])
            # If no chunks found in passage index (e.g. index not loaded), fall back to anchor's own passage_text
            if not chunks:
                matched_anchor_meta = next(
                    (m for m, _ in query_anchor_results if str(m.get("passage_id") or "") == pid), None
                )
                if matched_anchor_meta:
                    chunks = [{
                        "chunk_id": matched_anchor_meta.get("anchor_id", f"{pid}_qa"),
                        "passage_id": pid,
                        "text": matched_anchor_meta.get("passage_text", matched_anchor_meta.get("text", "")),
                        "raw_text": matched_anchor_meta.get("passage_text", matched_anchor_meta.get("raw_text", "")),
                        "lang": matched_anchor_meta.get("lang", lang),
                        "metadata": matched_anchor_meta.get("metadata", {}),
                    }]

            for chunk_meta in chunks:
                doc_id = str(chunk_meta.get("chunk_id") or chunk_meta.get("passage_id") or pid)
                if doc_id not in doc_map:
                    doc_map[doc_id] = chunk_meta
                q2q_sims[doc_id] = max(q2q_sims.get(doc_id, 0.0), sim_score)
                rrf_score = self.query_anchor_weight * (1.0 / (self.rrf_k + rank))
                scores_map[doc_id] = scores_map.get(doc_id, 0.0) + rrf_score
                match_sources.setdefault(doc_id, []).append(f"query_anchor→passage_lookup (sim={sim_score:.3f})")

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

