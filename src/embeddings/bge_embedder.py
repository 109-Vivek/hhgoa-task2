import os
import time
from typing import List, Union, Dict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM

_global_embedder_instance = None


class BGEEmbedder:
    """
    High-performance multilingual dense embedding generator.
    Includes LRU query embedding cache, dynamic dimension resolution,
    and CPU/GPU inference optimization.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Optimize CPU threads for parallel inference
        if self.device == "cpu":
            num_cores = os.cpu_count() or 4
            try:
                torch.set_num_threads(max(1, min(num_cores, 8)))
            except Exception:
                pass

        print(f"[Embedder] Loading {model_name} on device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Dynamically resolve embedding dimension from model
        try:
            if hasattr(self.model, "get_embedding_dimension"):
                self.dim = self.model.get_embedding_dimension() or EMBEDDING_DIM
            elif hasattr(self.model, "get_sentence_embedding_dimension"):
                self.dim = self.model.get_sentence_embedding_dimension() or EMBEDDING_DIM
            else:
                self.dim = EMBEDDING_DIM
        except Exception:
            self.dim = EMBEDDING_DIM

        # Enable FP16 where supported for 2x faster inference
        if self.device in ["cuda", "mps"]:
            try:
                self.model.half()
            except Exception:
                pass
        
        self.model.eval()
        
        # Fast in-memory LRU cache for query embeddings (capacity: 2048 queries)
        self._query_cache: Dict[str, np.ndarray] = {}
        self._max_cache_size: int = 2048

        # Warmup forward pass
        try:
            with torch.inference_mode():
                _ = self.model.encode(["warmup query text"], normalize_embeddings=True, convert_to_numpy=True)
        except Exception:
            pass

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encodes text string or list of texts into L2-normalized float32 numpy embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        with torch.inference_mode():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a single query string for vector search with microsecond LRU caching.
        """
        cache_key = query.strip()
        if cache_key in self._query_cache:
            return self._query_cache[cache_key].copy()

        vec = self.encode([query], normalize=True)[0]

        # Manage LRU cache size
        if len(self._query_cache) >= self._max_cache_size:
            # Pop oldest 20% entries
            keys_to_remove = list(self._query_cache.keys())[: self._max_cache_size // 5]
            for k in keys_to_remove:
                self._query_cache.pop(k, None)

        self._query_cache[cache_key] = vec
        return vec


def get_embedder() -> BGEEmbedder:
    """Singleton getter for the shared embedder instance."""
    global _global_embedder_instance
    if _global_embedder_instance is None:
        _global_embedder_instance = BGEEmbedder()
    return _global_embedder_instance

