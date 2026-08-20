import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Allow running directly as `python src\indexer.py` from repo root on Windows
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.config import (
    INDEX_DIR,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANG,
    FORCE_SAMPLE_CORPUS,
    ALLOW_DATASET_FALLBACK,
)
from src.chunking.chunker import MultiTierChunkingEngine, ChunkingStrategy, Chunk
from src.embeddings.bge_embedder import get_embedder, BGEEmbedder
from src.indexing.dense_index import FAISSIndex
from src.indexing.bm25_index import BM25Index

# Curated High-Quality Multilingual Indic Sample Passages for Instant Offline Indexing
SAMPLE_CORPUS = {
    "gu": [
        {
            "passage_id": "gu_101",
            "query": "ભારતની રાજધાની કઈ છે?",
            "passage": "નવી દિલ્હી ભારતની સત્તાવાર રાજધાની છે અને ભારત સરકારની ત્રણેય શાખાઓનું કેન્દ્ર છે. આ શહેરનો શિલાન્યાસ ૧૯૧૧ ના દિલ્હી દરબાર દરમિયાન જ્યોર્જ પાંચમા દ્વારા કરવામાં આવ્યો હતો.",
        },
        {
            "passage_id": "gu_102",
            "query": "ભારતના વડાપ્રધાન કોણ છે?",
            "passage": "ભારતના વડાપ્રધાન ભારત સરકારના વડા છે. નરેન્દ્ર મોદી મે ૨૦૧૪ થી ભારતના ૧૪મા વડાપ્રધાન તરીકે સેવા આપી રહ્યા છે.",
        },
        {
            "passage_id": "gu_103",
            "query": "ગોવા ક્યાં આવેલું છે?",
            "passage": "ગોવા ભારતનાં દક્ષિણ-પશ્ચિમ કિનારે કોંકણ પ્રદેશમાં આવેલું એક રાજ્ય છે. તે વિસ્તારની દ્રષ્ટિએ ભારતનું સૌથી નાનું રાજ્ય છે અને તેના સુંદર દરિયાકિનારા માટે જાણીતું છે.",
        },
        {
            "passage_id": "gu_104",
            "query": "રીટ્રીવલ-ઓગમેન્ટેડ જનરેશન (RAG) શું છે?",
            "passage": "રીટ્રીવલ-ઓગમેન્ટેડ જનરેશન (RAG) એ એક અદ્યતન AI તકનીક છે જે બાહ્ય ડેટાબેઝમાંથી સચોટ માહિતી મેળવીને ભાષા મોડેલોને સચોટ જવાબો આપવા સક્ષમ બનાવે છે, જેથી ભ્રમણા (hallucination) ઘટે છે.",
        },
        {
            "passage_id": "gu_105",
            "query": "વિશ્વનો સૌથી મોટો મહાસાગર કયો છે?",
            "passage": "પ્રશાંત મહાસાગર પૃથ્વીના પાંચ મહાસાગરોમાં સૌથી મોટો અને સૌથી ઊંડો છે. તે ઉત્તરમાં આર્કટિક મહાસાગરથી દક્ષિણમાં દક્ષિણી મહાસાગર સુધી વિસ્તરેલો છે.",
        },
    ],
    "hi": [
        {
            "passage_id": "hi_201",
            "query": "भारत की राजधानी क्या है?",
            "passage": "नई दिल्ली भारत की राजधानी है और भारत सरकार की तीनों शाखाओं (कार्यपालिका, विधायिका और न्यायपालिका) का केंद्र है। 1911 के दिल्ली दरबार के दौरान जॉर्ज पंचम द्वारा इस शहर की आधारशिला रखी गई थी।",
        },
        {
            "passage_id": "hi_202",
            "query": "भारत के प्रधानमंत्री कौन हैं?",
            "passage": "भारत के प्रधानमंत्री भारत सरकार के मुखिया होते हैं। नरेंद्र मोदी मई 2014 से भारत के 14वें प्रधानमंत्री के रूप में कार्यरत हैं।",
        },
        {
            "passage_id": "hi_203",
            "query": "गोवा कहाँ स्थित है?",
            "passage": "गोवा भारत के दक्षिण-पश्चिमी तट पर कोंकण क्षेत्र में स्थित एक राज्य है। यह क्षेत्रफल के हिसाब से भारत का सबसे छोटा राज्य है और अपनी समृद्ध संस्कृति और समुद्र तटों के लिए जाना जाता है।",
        },
        {
            "passage_id": "hi_204",
            "query": "रिट्रीवल-ऑगमेंटेड जनरेशन क्या है?",
            "passage": "रिट्रीवल-ऑगमेंटेड जनरेशन (RAG) एक आधुनिक एआई तकनीक है जो भाषा मॉडलों को बाहरी ज्ञानकोश से सटीक संदर्भ खोजकर तथ्य-आधारित उत्तर देने में सक्षम बनाती है, जिससे गलत जानकारी (hallucination) कम होती है।",
        },
        {
            "passage_id": "hi_205",
            "query": "विश्व का सबसे बड़ा महासागर कौन सा है?",
            "passage": "प्रशांत महासागर पृथ्वी का सबसे बड़ा और सबसे गहरा महासागर है। यह उत्तर में आर्कटिक महासागर से लेकर दक्षिण में दक्षिणी महासागर तक फैला हुआ है।",
        },
    ],
    "te": [
        {
            "passage_id": "te_301",
            "query": "భారతదేశ రాజధాని ఏది?",
            "passage": "న్యూఢిల్లీ భారతదేశ అధికారిక రాజధాని మరియు భారత ప్రభుత్వ మూడు శాఖల కేంద్రం. 1911 ఢిల్లీ దర్బార్ సందర్భంగా ఐదవ జార్జ్ రాజు ఈ నగరానికి శంకుస్థాపన చేశారు.",
        },
        {
            "passage_id": "te_302",
            "query": "భారత ప్రధానమంత్రి ఎవరు?",
            "passage": "భారత ప్రధానమంత్రి భారత ప్రభుత్వానికి అధిపతి. నరేంద్ర మోదీ మే 2014 నుండి భారతదేశ 14వ ప్రధానమంత్రిగా పనిచేస్తున్నారు.",
        },
        {
            "passage_id": "te_303",
            "query": "గోవా ఎక్కడ ఉంది?",
            "passage": "గోవా భారతదేశ నైరుతి తీరంలో కొంకణ్ ప్రాంతంలో ఉన్న ఒక రాష్ట్రం. విస్తీర్ణంలో ఇది భారతదేశంలోనే అతి చిన్న రాష్ట్రం మరియు అందమైన బీచ్‌లకు ప్రసిద్ధి చెందింది.",
        },
        {
            "passage_id": "te_304",
            "query": "రిట్రీవల్-ఆగ్మెంటెడ్ జనరేషన్ (RAG) అంటే ఏమిటి?",
            "passage": "RAG (Retrieval-Augmented Generation) అనేది బాహ్య సమాచార డేటాబేస్ నుండి ఖచ్చితమైన సందర్భాన్ని తిరిగి పొంది, AI మోడల్స్ ద్వారా నమ్మకమైన సమాధానాలను రూపొందించే అధునాతన సాంకేతికత.",
        },
        {
            "passage_id": "te_305",
            "query": "ప్రపంచంలో అతిపెద్ద మహాసముద్రం ఏది?",
            "passage": "పసిఫిక్ మహాసముద్రం భూమి యొక్క ఐదు మహాసముద్రాలలో అతిపెద్దది మరియు లోతైనది. ఇది ఉత్తరాన ఆర్కిటిక్ మహాసముద్రం నుండి దక్షిణాన దక్షిణ మహాసముద్రం వరకు విస్తరించి ఉంది.",
        },
    ],
}


def load_msmarco_xi_dataset(lang: str, limit: int = 500) -> List[Dict[str, Any]]:
    """
    Load language-specific samples from ai4bharat/MSMARCO-XI.

    Uses huggingface_hub to download the per-language parquet file and
    pyarrow's iter_batches() to read it. This avoids the
    ArrowNotImplementedError ("Nested data conversions not implemented
    for chunked array outputs") that occurs with pyarrow >= 15 when
    streaming nested struct columns via the ``datasets`` library.
    """
    if FORCE_SAMPLE_CORPUS:
        raise RuntimeError(
            "FORCE_SAMPLE_CORPUS=True, but fallback/sample mode is disabled. "
            "Unset FORCE_SAMPLE_CORPUS to use MSMARCO-XI."
        )

    # Map ISO 639-1 codes to the filename prefixes used in the repo.
    LANG_TO_FILE_PREFIX = {
        "gu": "guj",
        "hi": "hin",
        "te": "tel",
        "bn": "ben",
        "kn": "kan",
        "ml": "mal",
        "mr": "mar",
        "ne": "nep",
        "or": "ori",
        "pa": "pan",
        "ta": "tam",
        "ur": "urd",
        "as": "asm",
        "sa": "san",
    }

    prefix = LANG_TO_FILE_PREFIX.get(lang)
    if not prefix:
        raise RuntimeError(
            f"[Indexer] No MSMARCO-XI parquet mapping for language '{lang}'. "
            f"Supported: {', '.join(sorted(LANG_TO_FILE_PREFIX))}"
        )

    parquet_filename = f"train/{prefix}train.parquet"

    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        print(
            f"[Indexer] Loading Hugging Face dataset "
            f"'ai4bharat/MSMARCO-XI' file '{parquet_filename}' "
            f"for '{lang}' (limit: {limit})...",
            flush=True,
        )

        # Download (or use cached) language-specific parquet file
        local_path = hf_hub_download(
            repo_id="ai4bharat/MSMARCO-XI",
            filename=parquet_filename,
            repo_type="dataset",
        )

        # Read using iter_batches to get RecordBatch objects (plain
        # arrays, not chunked), which avoids the pyarrow nested-struct
        # chunked-array conversion bug.
        pf = pq.ParquetFile(local_path)

        passages: List[Dict[str, Any]] = []
        scanned_count = 0

        for batch in pf.iter_batches(batch_size=500):
            if len(passages) >= limit:
                break

            # Convert batch to list of dicts
            batch_rows = batch.to_pylist()

            for item in batch_rows:
                scanned_count += 1
                if len(passages) >= limit:
                    break

                query = str(item.get("query") or "").strip()

                # MSMARCO-XI stores passages as a nested structure.
                passage_data = item.get("passages")

                if not passage_data:
                    continue

                translated_passages = passage_data.get(
                    "Translated_passages", []
                )
                selected = passage_data.get("is_selected", [])

                if not translated_passages:
                    continue

                # Prefer selected passages, otherwise use all.
                selected_passages = []

                if selected and len(selected) == len(translated_passages):
                    selected_passages = [
                        text
                        for text, flag in zip(translated_passages, selected)
                        if flag and text
                    ]

                if not selected_passages:
                    selected_passages = [
                        text for text in translated_passages if text
                    ]

                for passage_idx, passage_text in enumerate(
                    selected_passages
                ):
                    if len(passages) >= limit:
                        break

                    passage_text = str(passage_text).strip()

                    if not passage_text:
                        continue

                    query_id = item.get("query_id", len(passages))

                    passages.append(
                        {
                            "passage_id": f"{lang}_{query_id}_{passage_idx}",
                            "passage": passage_text,
                            "query": query,
                        }
                    )

                    if len(passages) % 50 == 0 or len(passages) == limit:
                        print(
                            f"  [Dataset Loader] Loaded {len(passages)}/{limit} "
                            f"passages for '{lang}' "
                            f"(scanned {scanned_count} records)...",
                            flush=True,
                        )

        if not passages:
            raise RuntimeError(
                f"No usable passages found in MSMARCO-XI "
                f"for language '{lang}'."
            )

        print(
            f"[Indexer] Successfully loaded {len(passages)} "
            f"passages for '{lang}'.",
            flush=True,
        )

        return passages

    except Exception as e:
        # NO FALLBACK.
        raise RuntimeError(
            f"[Indexer] Failed to load MSMARCO-XI "
            f"for language '{lang}': {e}"
        ) from e


# Hardcoded Default Indexing Parameters
DEFAULT_INDEX_LANGUAGES = ["gu", "hi", "te"]
DEFAULT_MAX_SAMPLES = 500
DEFAULT_STRATEGY_NAME = "metadata_augmented"
INDEX_BATCH_SIZE = 50


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Loads indexing checkpoint if available."""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"processed_count": 0, "indexed_ids": [], "status": "none"}


def save_checkpoint(checkpoint_path: Path, data: Dict[str, Any]):
    """Persists indexing checkpoint to disk."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_indices_for_language(
    lang: str,
    limit: int = DEFAULT_MAX_SAMPLES,
    strategy: ChunkingStrategy = ChunkingStrategy.METADATA_AUGMENTED,
    save_dir: Path = INDEX_DIR,
    embedder: Optional[BGEEmbedder] = None,
    use_sample: bool = False,
):
    """
    Builds dense (FAISS HNSW) and lexical (BM25s) indices for a specific language.
    Supports incremental resumption from saved checkpoints.
    """
    print(f"\n=======================================================")
    print(f"[Indexer] Starting Indexing Pipeline for: [{lang.upper()}]")
    print(f"[Indexer] Strategy: {strategy.value} | Target Limit: {limit}")
    print(f"=======================================================")

    target_dir = Path(save_dir) / lang
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = target_dir / "checkpoint.json"

    # Check for existing checkpoint to resume
    checkpoint = load_checkpoint(checkpoint_file)
    processed_count = checkpoint.get("processed_count", 0)
    indexed_ids = set(checkpoint.get("indexed_ids", []))

    dense_index = FAISSIndex()
    query_dense_index = FAISSIndex()
    bm25_index = BM25Index()

    # If previous indices exist on disk, load them for incremental resumption
    if processed_count > 0 and (target_dir / "faiss.index").exists():
        dense_loaded = dense_index.load(target_dir, index_name="faiss.index", meta_name="faiss_meta.json")
        query_loaded = query_dense_index.load(target_dir, index_name="query_faiss.index", meta_name="query_faiss_meta.json")
        bm25_loaded = bm25_index.load(target_dir)

        if dense_loaded and bm25_loaded:
            print(f"[Indexer] [RESUME] Found existing checkpoint: {processed_count}/{limit} items already indexed for '{lang}'.")
            if processed_count >= limit:
                print(f"[Indexer] [COMPLETE] Target limit ({limit}) already reached for '{lang}'. Skipping.")
                return

    chunker = MultiTierChunkingEngine()
    embedder = embedder or get_embedder()

    # 1. Fetch raw data from stream (excluding already indexed IDs)
    remaining_needed = limit - processed_count
    if use_sample:
        raise RuntimeError(
            "--use-sample is disabled. This indexer must use MSMARCO-XI."
        )

    raw_items = load_msmarco_xi_dataset(lang, limit=remaining_needed)

    # Filter out already indexed items
    new_items = [item for item in raw_items if str(item.get("passage_id", item.get("id", ""))) not in indexed_ids]

    if not new_items and processed_count > 0:
        print(f"[Indexer] All available items ({processed_count}) already indexed for '{lang}'.")
        return

    total_batches = (len(new_items) + INDEX_BATCH_SIZE - 1) // INDEX_BATCH_SIZE
    print(f"[Indexer] Processing {len(new_items)} new passages for language '{lang}' in {total_batches} batches (batch_size={INDEX_BATCH_SIZE})...", flush=True)

    # Process in incremental batches
    for batch_idx, batch_start in enumerate(range(0, len(new_items), INDEX_BATCH_SIZE), start=1):
        batch = new_items[batch_start : batch_start + INDEX_BATCH_SIZE]
        print(f"\n--- [Batch {batch_idx}/{total_batches}] Processing {len(batch)} items for '{lang}' ---", flush=True)
        
        # 2. Chunking
        start_chunk = time.perf_counter()
        batch_chunks: List[Chunk] = []
        for item in batch:
            chunks = chunker.chunk_passage(
                passage_id=item["passage_id"],
                passage_text=item["passage"],
                lang=lang,
                query=item.get("query"),
                strategy=strategy,
            )
            batch_chunks.extend(chunks)
        chunk_duration = (time.perf_counter() - start_chunk) * 1000.0
        print(f"  [1/5 Chunking] Created {len(batch_chunks)} chunks ({chunk_duration:.1f}ms)", flush=True)

        if not batch_chunks:
            print("  [Notice] No chunks generated for this batch, skipping.", flush=True)
            continue

        # 3. Dense Embedding (Passages)
        texts_to_embed = [c.text for c in batch_chunks]
        start_embed = time.perf_counter()
        embeddings = embedder.encode(texts_to_embed, batch_size=32, show_progress_bar=False)
        embed_duration = (time.perf_counter() - start_embed) * 1000.0

        metadata_list = [
            {
                "chunk_id": c.chunk_id,
                "passage_id": c.doc_id,
                "text": c.text,
                "raw_text": c.raw_text,
                "lang": c.lang,
                "strategy": c.strategy.value,
                "metadata": c.metadata,
            }
            for c in batch_chunks
        ]
        dense_index.add(embeddings, metadata_list)
        print(f"  [2/5 Dense Embeddings] Encoded & added {len(texts_to_embed)} vectors ({embed_duration:.1f}ms, total in index: {dense_index.count()})", flush=True)

        # 4. Query-Anchor Dense Indexing (Dual-Track)
        start_qa = time.perf_counter()
        query_anchors = chunker.extract_query_anchors(batch, lang=lang)
        if query_anchors:
            queries_to_embed = [qa.query for qa in query_anchors]
            query_embeddings = embedder.encode(queries_to_embed, batch_size=32, show_progress_bar=False)
            query_metadata_list = [
                {
                    "anchor_id": qa.anchor_id,
                    "passage_id": qa.passage_id,
                    "query": qa.query,
                    "passage_text": qa.passage_text,
                    "raw_text": qa.passage_text,
                    "lang": qa.lang,
                    "metadata": qa.metadata,
                }
                for qa in query_anchors
            ]
            query_dense_index.add(query_embeddings, query_metadata_list)
            qa_duration = (time.perf_counter() - start_qa) * 1000.0
            print(f"  [3/5 Query Anchors] Encoded & added {len(queries_to_embed)} query anchors ({qa_duration:.1f}ms, total query index: {query_dense_index.count()})", flush=True)
        else:
            print(f"  [3/5 Query Anchors] No query anchors present in this batch", flush=True)

        # 5. BM25 Re-indexing with updated corpus
        start_bm25 = time.perf_counter()
        all_passage_texts = [doc.get("text", "") for doc in dense_index.metadata_store]
        bm25_index.index_documents(all_passage_texts, dense_index.metadata_store)
        bm25_duration = (time.perf_counter() - start_bm25) * 1000.0
        print(f"  [4/5 BM25 Sparse] Re-indexed {len(all_passage_texts)} documents ({bm25_duration:.1f}ms)", flush=True)

        # Update checkpoint tracking
        for item in batch:
            p_id = str(item.get("passage_id", item.get("id", "")))
            if p_id:
                indexed_ids.add(p_id)

        current_total = len(dense_index.metadata_store)

        # 6. Save intermediate progress to disk
        start_save = time.perf_counter()
        dense_index.save(target_dir, index_name="faiss.index", meta_name="faiss_meta.json")
        if query_dense_index.count() > 0:
            query_dense_index.save(target_dir, index_name="query_faiss.index", meta_name="query_faiss_meta.json")
        bm25_index.save(target_dir)

        save_checkpoint(
            checkpoint_file,
            {
                "lang": lang,
                "processed_count": current_total,
                "target_limit": limit,
                "last_updated": time.time(),
                "status": "completed" if current_total >= limit else "in_progress",
                "indexed_ids": list(indexed_ids),
            },
        )
        save_duration = (time.perf_counter() - start_save) * 1000.0
        print(f"  [5/5 Checkpoint & Disk Save] Saved indices to {target_dir} ({save_duration:.1f}ms) -> [{current_total}/{limit} total]", flush=True)

    print(f"\n[Indexer] [SUCCESS] Completed indexing {dense_index.count()} passages & {query_dense_index.count()} query anchors for '{lang}'.\n", flush=True)


def build_all_indices(
    languages: List[str] = DEFAULT_INDEX_LANGUAGES,
    limit: int = DEFAULT_MAX_SAMPLES,
    strategy_name: str = DEFAULT_STRATEGY_NAME,
    save_dir: Path = INDEX_DIR,
    use_sample: bool = False,
):
    strategy_map = {
        "atomic_passage": ChunkingStrategy.ATOMIC_PASSAGE,
        "sliding_window": ChunkingStrategy.SLIDING_WINDOW,
        "metadata_augmented": ChunkingStrategy.METADATA_AUGMENTED,
        "query_anchor": ChunkingStrategy.QUERY_ANCHOR,
    }
    strategy = strategy_map.get(strategy_name, ChunkingStrategy.METADATA_AUGMENTED)

    embedder = get_embedder()
    for lang in languages:
        build_indices_for_language(
            lang=lang,
            limit=limit,
            strategy=strategy,
            save_dir=save_dir,
            embedder=embedder,
            use_sample=use_sample,
        )


def main():
    parser = argparse.ArgumentParser(description="Voice Indic RAG Resumable Indexer for MSMARCO-XI")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=DEFAULT_INDEX_LANGUAGES,
        help=f"List of language codes to index (default: {' '.join(DEFAULT_INDEX_LANGUAGES)})",
    )
    parser.add_argument(
        "--limit",
        "--max-samples",
        dest="limit",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Number of passages to index per language (default: {DEFAULT_MAX_SAMPLES})",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_STRATEGY_NAME,
        choices=["atomic_passage", "sliding_window", "metadata_augmented", "query_anchor"],
        help=f"Chunking strategy to use (default: {DEFAULT_STRATEGY_NAME})",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Force using built-in Indic sample passages instead of live stream",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=str(INDEX_DIR),
        help="Directory to save the built indices",
    )

    args = parser.parse_args()
    build_all_indices(
        languages=args.languages,
        limit=args.limit,
        strategy_name=args.strategy,
        save_dir=Path(args.index_dir),
        use_sample=args.use_sample,
    )


if __name__ == "__main__":
    main()
