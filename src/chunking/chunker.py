import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ChunkingStrategy(str, Enum):
    ATOMIC_PASSAGE = "atomic_passage"
    SLIDING_WINDOW = "sliding_window"
    METADATA_AUGMENTED = "metadata_augmented"
    QUERY_ANCHOR = "query_anchor"


@dataclass
class Chunk:
    chunk_id: str
    text: str
    raw_text: str
    doc_id: str
    lang: str
    strategy: ChunkingStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiTierChunkingEngine:
    """
    A multi-tiered, metadata-aware chunking engine designed for MSMARCO-XI.
    Supports atomic passage chunking, sliding-window overlap splitting,
    metadata augmentation, and query-anchor pairing.
    """

    def __init__(
        self,
        default_chunk_size: int = 256,
        default_overlap: int = 64,
    ):
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap

    @staticmethod
    def _split_into_sentences(text: str, lang: str = "en") -> List[str]:
        if not text:
            return []
        pattern = r"(?<=[.!?।\|])\s+"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_passage(
        self,
        passage_id: str,
        passage_text: str,
        lang: str = "en",
        query: Optional[str] = None,
        strategy: ChunkingStrategy = ChunkingStrategy.METADATA_AUGMENTED,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        meta = extra_metadata or {}
        meta["lang"] = lang
        meta["passage_id"] = passage_id

        if strategy == ChunkingStrategy.ATOMIC_PASSAGE:
            return [
                Chunk(
                    chunk_id=f"{passage_id}_atomic",
                    text=passage_text,
                    raw_text=passage_text,
                    doc_id=passage_id,
                    lang=lang,
                    strategy=strategy,
                    metadata=meta,
                )
            ]

        elif strategy == ChunkingStrategy.METADATA_AUGMENTED:
            augmented_text = f"[LANG: {lang}] [DOC: {passage_id}] {passage_text}"
            return [
                Chunk(
                    chunk_id=f"{passage_id}_meta",
                    text=augmented_text,
                    raw_text=passage_text,
                    doc_id=passage_id,
                    lang=lang,
                    strategy=strategy,
                    metadata=meta,
                )
            ]

        elif strategy == ChunkingStrategy.QUERY_ANCHOR:
            anchor_text = f"Query: {query}\nPassage: {passage_text}" if query else passage_text
            meta["query_anchor"] = query
            return [
                Chunk(
                    chunk_id=f"{passage_id}_anchor",
                    text=anchor_text,
                    raw_text=passage_text,
                    doc_id=passage_id,
                    lang=lang,
                    strategy=strategy,
                    metadata=meta,
                )
            ]

        elif strategy == ChunkingStrategy.SLIDING_WINDOW:
            sentences = self._split_into_sentences(passage_text, lang=lang)
            chunks = []
            words = passage_text.split()
            if len(words) <= self.default_chunk_size:
                return [
                    Chunk(
                        chunk_id=f"{passage_id}_sw_0",
                        text=passage_text,
                        raw_text=passage_text,
                        doc_id=passage_id,
                        lang=lang,
                        strategy=strategy,
                        metadata=meta,
                    )
                ]

            current_chunk_words = []
            chunk_idx = 0

            for sent in sentences:
                sent_words = sent.split()
                if len(current_chunk_words) + len(sent_words) > self.default_chunk_size and current_chunk_words:
                    chunk_str = " ".join(current_chunk_words)
                    chunks.append(
                        Chunk(
                            chunk_id=f"{passage_id}_sw_{chunk_idx}",
                            text=f"[LANG: {lang}] {chunk_str}",
                            raw_text=chunk_str,
                            doc_id=passage_id,
                            lang=lang,
                            strategy=strategy,
                            metadata={**meta, "chunk_index": chunk_idx},
                        )
                    )
                    chunk_idx += 1
                    overlap_size = min(self.default_overlap, len(current_chunk_words))
                    current_chunk_words = current_chunk_words[-overlap_size:]

                current_chunk_words.extend(sent_words)

            if current_chunk_words:
                chunk_str = " ".join(current_chunk_words)
                chunks.append(
                    Chunk(
                        chunk_id=f"{passage_id}_sw_{chunk_idx}",
                        text=f"[LANG: {lang}] {chunk_str}",
                        raw_text=chunk_str,
                        doc_id=passage_id,
                        lang=lang,
                        strategy=strategy,
                        metadata={**meta, "chunk_index": chunk_idx},
                    )
                )

            return chunks

        return []
