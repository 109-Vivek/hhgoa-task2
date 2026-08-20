import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

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
    "en": [
        {
            "passage_id": "en_101",
            "query": "What is the capital of India?",
            "passage": "New Delhi is the official capital of India and the seat of all three branches of the Government of India. The foundation stone of the city was laid by George V during the 1911 Delhi Durbar.",
        },
        {
            "passage_id": "en_102",
            "query": "Who is the Prime Minister of India?",
            "passage": "The Prime Minister of India is the head of the government of India. Narendra Modi has served as the 14th Prime Minister of India since May 2014.",
        },
        {
            "passage_id": "en_103",
            "query": "What is Retrieval-Augmented Generation?",
            "passage": "Retrieval-Augmented Generation (RAG) is an AI architecture that enhances large language models by retrieving authoritative external knowledge bases before generating a response, drastically reducing hallucinations.",
        },
        {
            "passage_id": "en_104",
            "query": "What is the largest ocean in the world?",
            "passage": "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions. It extends from the Arctic Ocean in the north to the Southern Ocean in the south.",
        },
        {
            "passage_id": "en_105",
            "query": "Where is Goa located?",
            "passage": "Goa is a state on the southwestern coast of India within the region known as the Konkan. It is bounded by the state of Maharashtra to the north and by Karnataka to the east and south, with the Arabian Sea forming its western coast. It is India's smallest state by area.",
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
    "ta": [
        {
            "passage_id": "ta_301",
            "query": "இந்தியாவின் தலைநகரம் எது?",
            "passage": "புது தில்லி இந்தியாவின் தலைநகரமாகும். இது இந்திய அரசாங்கத்தின் மூன்று கிளைகளின் மையமாக விளங்குகிறது. 1911 ஆம் ஆண்டு தில்லி தர்பாரின் போது ஐந்தாம் ஜார்ஜ் மன்னரால் இந்நகரத்திற்கான அடிக்கல் நாட்டப்பட்டது.",
        },
        {
            "passage_id": "ta_302",
            "query": "இந்தியாவின் பிரதமர் யார்?",
            "passage": "இந்தியப் பிரதமர் இந்திய அரசாங்கத்தின் தலைவர் ஆவார். நரேந்திர மோடி மே 2014 முதல் இந்தியாவின் 14வது பிரதமராகப் பணியாற்றி வருகிறார்.",
        },
        {
            "passage_id": "ta_303",
            "query": "கோவா எங்கு அமைந்துள்ளது?",
            "passage": "கோவா இந்தியாவின் தென்மேற்கு கடற்கரையில் அமைந்துள்ள ஒரு மாநிலமாகும். பரப்பளவில் இது இந்தியாவின் மிகச்சிறிய மாநிலமாகும், மேலும் இது அழகான கடற்கரைகளுக்குப் புகழ்பெற்றது.",
        },
        {
            "passage_id": "ta_304",
            "query": "மீட்டெடுப்பு ஆக்மென்டட் தலைமுறை (RAG) என்றால் என்ன?",
            "passage": "RAG (Retrieval-Augmented Generation) என்பது வெளிப்புறத் தரவுத்தளங்களிலிருந்து துல்லியமான தகவல்களை மீட்டெடுத்து செயற்கை நுண்ணறிவு மாதிரிகள் மூலம் நம்பகமான பதில்களை உருவாக்கும் ஒரு நவீன நுட்பமாகும்.",
        },
        {
            "passage_id": "ta_305",
            "query": "உலகின் மிகப்பெரிய பெருங்கடல் எது?",
            "passage": "பசிபிக் பெருங்கடல் பூமியின் ஐந்து பெருங்கடல்களில் மிகப்பெரியதும் ஆழமானதுமாகும். இது வடக்கில் ஆர்க்டிக் பெருங்கடலில் இருந்து தெற்கில் தென் பெருங்கடல் வரை பரவியுள்ளது.",
        },
    ],
}


def load_msmarco_xi_dataset(lang: str, limit: int = 500) -> List[Dict[str, Any]]:
    """
    Attempts to load passages from Hugging Face ai4bharat/MSMARCO-XI dataset.
    Respects FORCE_SAMPLE_CORPUS and ALLOW_DATASET_FALLBACK environment toggles.
    """
    if FORCE_SAMPLE_CORPUS:
        print(f"[Indexer] FORCE_SAMPLE_CORPUS=True is active. Using curated sample corpus for '{lang}'.")
        return SAMPLE_CORPUS.get(lang, SAMPLE_CORPUS["en"])

    try:
        from datasets import load_dataset
        print(f"[Indexer] Attempting to load Hugging Face dataset 'ai4bharat/MSMARCO-XI' ({lang})...")
        # Load dataset stream or split
        dataset = load_dataset(
            "ai4bharat/MSMARCO-XI",
            lang,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        
        passages = []
        for i, item in enumerate(dataset):
            if i >= limit:
                break
            p_id = str(item.get("passage_id", item.get("id", f"{lang}_{i}")))
            p_text = item.get("passage", item.get("text", ""))
            query = item.get("query", "")
            if p_text.strip():
                passages.append({
                    "passage_id": p_id,
                    "passage": p_text,
                    "query": query,
                })
        
        if passages:
            print(f"[Indexer] Successfully loaded {len(passages)} passages from Hugging Face for '{lang}'.")
            return passages

    except Exception as e:
        err_msg = f"Could not load Hugging Face dataset ({e})"
        print(f"[Indexer] {err_msg}")
        if not ALLOW_DATASET_FALLBACK:
            raise RuntimeError(f"[Indexer] {err_msg} and ALLOW_DATASET_FALLBACK=False")

    # Fallback to high-quality curated dataset
    print(f"[Indexer] Falling back to curated sample corpus for '{lang}'.")
    return SAMPLE_CORPUS.get(lang, SAMPLE_CORPUS["en"])


def build_indices_for_language(
    lang: str,
    limit: int = 100,
    strategy: ChunkingStrategy = ChunkingStrategy.METADATA_AUGMENTED,
    save_dir: Path = INDEX_DIR,
    embedder: Optional[BGEEmbedder] = None,
    use_sample: bool = False,
):
    """
    Builds dense (FAISS HNSW) and lexical (BM25s) indices for a specific language.
    """
    print(f"\n=======================================================")
    print(f"[Indexer] Starting Indexing Pipeline for: [{lang.upper()}]")
    print(f"[Indexer] Strategy: {strategy.value} | Limit: {limit}")
    print(f"=======================================================")

    chunker = MultiTierChunkingEngine()
    embedder = embedder or get_embedder()

    # 1. Fetch raw data
    if use_sample or FORCE_SAMPLE_CORPUS:
        raw_items = SAMPLE_CORPUS.get(lang, SAMPLE_CORPUS["en"])
    else:
        raw_items = load_msmarco_xi_dataset(lang, limit=limit)

    # 2. Chunking
    all_chunks: List[Chunk] = []
    for item in raw_items:
        chunks = chunker.chunk_passage(
            passage_id=item["passage_id"],
            passage_text=item["passage"],
            lang=lang,
            query=item.get("query"),
            strategy=strategy,
        )
        all_chunks.extend(chunks)

    print(f"[Indexer] Generated {len(all_chunks)} chunks from {len(raw_items)} source passages.")

    # 3. Dense Embedding
    texts_to_embed = [c.text for c in all_chunks]
    print(f"[Indexer] Computing BGE-M3 Dense Embeddings ({len(texts_to_embed)} vectors)...")
    start_embed = time.perf_counter()
    embeddings = embedder.encode(texts_to_embed, batch_size=32, show_progress_bar=True)
    embed_duration = (time.perf_counter() - start_embed) * 1000.0
    print(f"[Indexer] Dense embedding computed in {embed_duration:.2f} ms ({embed_duration/len(texts_to_embed):.2f} ms/vector)")

    # Prepare metadata
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
        for c in all_chunks
    ]

    # 4. Dense FAISS HNSW Indexing
    dense_index = FAISSIndex()
    dense_index.add(embeddings, metadata_list)

    # 5. Lexical BM25s Indexing
    bm25_index = BM25Index()
    bm25_index.index_documents(texts_to_embed, metadata_list)

    # 6. Persistence
    target_dir = Path(save_dir) / lang
    print(f"[Indexer] Saving indices to disk at: {target_dir}")
    dense_index.save(target_dir)
    bm25_index.save(target_dir)

    print(f"[Indexer] [SUCCESS] Indexed {len(all_chunks)} chunks for language '{lang}' successfully!\n")


def build_all_indices(
    languages: List[str] = SUPPORTED_LANGUAGES,
    limit: int = 100,
    strategy_name: str = "metadata_augmented",
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
    parser = argparse.ArgumentParser(description="Voice Indic RAG Indexer for MSMARCO-XI")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "hi", "ta"],
        help="List of language codes to index (e.g. en hi ta)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of passages to index per language",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="metadata_augmented",
        choices=["atomic_passage", "sliding_window", "metadata_augmented", "query_anchor"],
        help="Chunking strategy to use",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Force using built-in Indic sample passages",
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
