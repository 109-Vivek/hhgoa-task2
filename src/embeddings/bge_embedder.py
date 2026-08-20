import time
import numpy as np
import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM

_global_embedder_instance = None


class BGEEmbedder:
    """
    High-performance multilingual dense embedding generator using BAAI/bge-m3.
    L2-normalized embeddings enable sub-millisecond cosine similarity via FAISS inner product.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"[Embedder] Loading {model_name} on device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        # Enable FP16 where supported for 2x faster inference
        if self.device in ["cuda", "mps"]:
            try:
                self.model.half()
            except Exception:
                pass
        
        self.dim = EMBEDDING_DIM

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

        start_time = time.perf_counter()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a single query string for vector search.
        """
        return self.encode([query], normalize=True)[0]


def get_embedder() -> BGEEmbedder:
    """Singleton getter for the shared embedder instance."""
    global _global_embedder_instance
    if _global_embedder_instance is None:
        _global_embedder_instance = BGEEmbedder()
    return _global_embedder_instance
