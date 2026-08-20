import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import bm25s


class BM25Index:
    """
    Ultra-fast C/Array-accelerated BM25 indexer using `bm25s` for lexical retrieval.
    """

    def __init__(self):
        self.retriever = bm25s.BM25()
        self.corpus_documents: List[str] = []
        self.metadata_store: List[Dict[str, Any]] = []
        self.is_indexed = False

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Multilingual tokenizer supporting Indic scripts (Gujarati, Devanagari/Hindi, Telugu) and alphanumeric tokens.
        """
        if not text:
            return []
        text = text.lower()
        # Keep alphanumeric, Devanagari (\u0900-\u097F), Gujarati (\u0A80-\u0AFF), Telugu (\u0C00-\u0C7F)
        tokens = re.findall(r"[\w\u0900-\u097F\u0A80-\u0AFF\u0C00-\u0C7F]+", text)
        return tokens

    def index_documents(self, documents: List[str], metadata: List[Dict[str, Any]]):
        """
        Tokenizes and builds the BM25 index.
        """
        if not documents:
            return

        self.corpus_documents = documents
        self.metadata_store = metadata

        corpus_tokens = bm25s.tokenize(documents, stopwords=None)
        self.retriever.index(corpus_tokens)
        self.is_indexed = True

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Executes BM25 search over query. Returns list of (metadata_dict, bm25_score).
        """
        if not self.is_indexed or len(self.corpus_documents) == 0:
            return []

        query_tokens = bm25s.tokenize(query)
        results, scores = self.retriever.retrieve(query_tokens, k=min(top_k, len(self.corpus_documents)))

        output = []
        for doc_item, score in zip(results[0], scores[0]):
            if isinstance(doc_item, dict):
                idx = int(doc_item.get("id", -1))
            else:
                try:
                    idx = int(doc_item)
                except Exception:
                    idx = -1

            if 0 <= idx < len(self.metadata_store):
                output.append((self.metadata_store[idx], float(score)))

        return output

    def save(self, save_dir: Path):
        """Saves BM25 index and corpus metadata."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        bm25_dir = save_dir / "bm25s_model"
        meta_path = save_dir / "bm25_meta.json"

        self.retriever.save(str(bm25_dir), corpus=self.corpus_documents)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)

    def load(self, save_dir: Path) -> bool:
        """Loads BM25 index from directory."""
        save_dir = Path(save_dir)
        bm25_dir = save_dir / "bm25s_model"
        meta_path = save_dir / "bm25_meta.json"

        if not bm25_dir.exists() or not meta_path.exists():
            return False

        try:
            self.retriever = bm25s.BM25.load(str(bm25_dir), load_corpus=True)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata_store = json.load(f)
            self.corpus_documents = self.retriever.corpus if hasattr(self.retriever, "corpus") else []
            self.is_indexed = True
            return True
        except Exception as e:
            print(f"[BM25] Load failed: {e}")
            return False
