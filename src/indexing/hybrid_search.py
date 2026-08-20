import time
from typing import List, Dict, Any, Tuple
from src.indexing.dense_index import FAISSIndex
from src.indexing.bm25_index import BM25Index
from src.embeddings.bge_embedder import BGEEmbedder, get_embedder
from src.config import RRF_K, DENSE_WEIGHT, LEXICAL_WEIGHT, SIMILARITY_THRESHOLD, MAX_RETRIEVAL_RESULTS


class HybridSearchEngine:
    """
    Sub-millisecond Hybrid Retrieval Engine combining FAISS HNSW (Dense) and BM25s (Lexical)
    using Reciprocal Rank Fusion (RRF) and dynamic score normalization.
    """

    def __init__(
        self,
        dense_index: FAISSIndex,
        lexical_index: BM25Index,
        embedder: BGEEmbedder = None,
        rrf_k: int = RRF_K,
        dense_weight: float = DENSE_WEIGHT,
        lexical_weight: float = LEXICAL_WEIGHT,
    ):
        self.dense_index = dense_index
        self.lexical_index = lexical_index
        self.embedder = embedder or get_embedder()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def search(
        self,
        query: str,
        top_k: int = MAX_RETRIEVAL_RESULTS,
        lang: str = "en",
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> Tuple[List[Dict[str, Any]], float, Dict[str, float]]:
        """
        Executes parallel dense + lexical search and reciprocal rank fusion.
        Returns: (fused_results, retrieval_time_ms, detailed_metrics)
        """
        start_time = time.perf_counter()
        timing = {}

        # 1. Query Embedding
        embed_start = time.perf_counter()
        query_vec = self.embedder.encode_query(query)
        timing["embedding_ms"] = (time.perf_counter() - embed_start) * 1000.0

        # 2. Dense FAISS Search
        dense_start = time.perf_counter()
        dense_results = self.dense_index.search(query_vec, top_k=top_k * 2)
        timing["dense_search_ms"] = (time.perf_counter() - dense_start) * 1000.0

        # 3. Lexical BM25 Search
        lexical_start = time.perf_counter()
        lexical_results = self.lexical_index.search(query, top_k=top_k * 2)
        timing["lexical_search_ms"] = (time.perf_counter() - lexical_start) * 1000.0

        # 4. Reciprocal Rank Fusion (RRF)
        fusion_start = time.perf_counter()
        doc_map: Dict[str, Dict[str, Any]] = {}
        scores_map: Dict[str, float] = {}
        dense_sims: Dict[str, float] = {}

        for rank, (meta, score) in enumerate(dense_results, start=1):
            doc_id = meta.get("passage_id") or meta.get("chunk_id")
            doc_map[doc_id] = meta
            dense_sims[doc_id] = score
            rrf_score = self.dense_weight * (1.0 / (self.rrf_k + rank))
            scores_map[doc_id] = scores_map.get(doc_id, 0.0) + rrf_score

        for rank, (meta, score) in enumerate(lexical_results, start=1):
            doc_id = meta.get("passage_id") or meta.get("chunk_id")
            doc_map[doc_id] = meta
            rrf_score = self.lexical_weight * (1.0 / (self.rrf_k + rank))
            scores_map[doc_id] = scores_map.get(doc_id, 0.0) + rrf_score

        timing["fusion_ms"] = (time.perf_counter() - fusion_start) * 1000.0

        # Sort documents by fused RRF score
        sorted_docs = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            meta = doc_map[doc_id]
            dense_score = dense_sims.get(doc_id, 0.0)
            
            item = {
                **meta,
                "rrf_score": rrf_score,
                "dense_score": dense_score,
            }
            final_results.append(item)

        total_retrieval_ms = (time.perf_counter() - start_time) * 1000.0
        timing["total_retrieval_ms"] = total_retrieval_ms

        return final_results, total_retrieval_ms, timing
