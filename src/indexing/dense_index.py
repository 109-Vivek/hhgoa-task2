import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from src.config import EMBEDDING_DIM


class FAISSIndex:
    """
    Sub-millisecond FAISS HNSW / Inner Product Vector Index for Dense Semantic Search.
    """

    def __init__(self, dim: int = EMBEDDING_DIM, use_hnsw: bool = True):
        self.dim = dim
        self.use_hnsw = use_hnsw
        if use_hnsw:
            # M=32 graph connections for high recall & sub-millisecond query time
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = 64
            self.index.hnsw.efConstruction = 128
        else:
            self.index = faiss.IndexFlatIP(dim)

        self.metadata_store: List[Dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Adds normalized vector embeddings and corresponding passage metadata."""
        if len(embeddings) == 0:
            return

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)
        self.metadata_store.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Executes k-nearest neighbor search using inner product (cosine similarity on normalized vectors).
        Returns list of (metadata_dict, similarity_score).
        """
        if self.index.ntotal == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx < len(self.metadata_store):
                results.append((self.metadata_store[idx], float(score)))

        return results

    def save(self, save_dir: Path, index_name: str = "faiss.index", meta_name: str = "faiss_meta.json"):
        """Persists the FAISS index and metadata store to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = save_dir / index_name
        meta_path = save_dir / meta_name

        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)

    def load(self, save_dir: Path, index_name: str = "faiss.index", meta_name: str = "faiss_meta.json") -> bool:
        """Loads FAISS index and metadata from disk."""
        save_dir = Path(save_dir)
        index_path = save_dir / index_name
        meta_path = save_dir / meta_name

        if not index_path.exists() or not meta_path.exists():
            return False

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata_store = json.load(f)
        return True

    def count(self) -> int:
        return self.index.ntotal
