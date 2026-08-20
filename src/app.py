import os
import sys
import time
import subprocess
from pathlib import Path
import streamlit as st

from src.config import (
    SUPPORTED_LANGUAGES,
    DEFAULT_LANG,
    INDEX_DIR,
    PRIMARY_LLM_PROVIDER,
    PRIMARY_LLM_MODEL,
    EMBEDDING_MODEL_NAME,
)
from src.harness.orchestrator import VoiceRAGOrchestrator, PipelineResponse
from src.indexer import build_all_indices, SAMPLE_CORPUS

# Page Config
st.set_page_config(
    page_title="HH Goa 2026 | Voice-Enabled Indic RAG",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4ECDC4;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaaaaa;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .guardrail-pass {
        color: #2ECC71;
        font-weight: 600;
    }
    .guardrail-blocked {
        color: #E74C3C;
        font-weight: 600;
    }
    .abstention-box {
        background: rgba(241, 196, 15, 0.1);
        border-left: 4px solid #f1c40f;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .context-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #45B7D1;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Indic RAG Orchestrator & BGE-M3 Embedder...")
def get_pipeline():
    # Auto-build indices if not present
    for lang in ["en", "hi", "ta"]:
        lang_dir = Path(INDEX_DIR) / lang
        if not (lang_dir / "faiss.index").exists():
            print(f"[App] Initializing index for {lang}...")
            build_all_indices(languages=[lang], limit=20, use_sample=True)
    return VoiceRAGOrchestrator()


def main():
    st.markdown('<div class="main-header">🎙️ Voice-Enabled Indic RAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">HH Goa 2026 Hackathon Task 2 • Multi-Indic (English, Hindi, Tamil) • Sub-200ms Latency Target</div>',
        unsafe_allow_html=True,
    )

    # Initialize Pipeline
    orchestrator = get_pipeline()

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info(f"**Embedding Model:** `{EMBEDDING_MODEL_NAME}`\n\n**Vector Search:** `FAISS HNSW (Inner Product)`\n\n**Lexical Search:** `BM25s (Indic-aware)`\n\n**STT Provider:** `Sarvam AI (saaras:v2)`")

        st.subheader("📚 Index Management")
        selected_lang_index = st.selectbox("Select Language to Re-Index", ["en", "hi", "ta"], index=0)
        chunking_strategy = st.selectbox(
            "Chunking Strategy",
            ["metadata_augmented", "atomic_passage", "sliding_window", "query_anchor"],
            index=0,
        )

        if st.button("🔨 Re-build Indices Now"):
            with st.spinner(f"Building {chunking_strategy} index for {selected_lang_index}..."):
                build_all_indices(
                    languages=[selected_lang_index],
                    limit=50,
                    strategy_name=chunking_strategy,
                    use_sample=True,
                )
                st.cache_resource.clear()
                st.success(f"Indices rebuilt for {selected_lang_index}!")
                st.rerun()

        st.divider()
        st.subheader("📊 Latency Benchmark")
        num_bench_queries = st.slider("Benchmark Queries", min_value=5, max_value=50, value=15, step=5)
        if st.button("⚡ Run Latency Benchmark"):
            with st.spinner("Running latency benchmark across en/hi/ta..."):
                from src.benchmark import run_benchmark
                res = run_benchmark(num_queries=num_bench_queries, languages=["en", "hi", "ta"])
                st.session_state["benchmark_results"] = res
                st.success("Benchmark completed! See Benchmark Tab.")

    # Main Tabs
    tab_rag, tab_benchmark, tab_specs = st.tabs(["🎙️ Voice / Text Query", "📈 Latency Analytics", "📐 System Architecture"])

    with tab_rag:
        col_input, col_lang = st.columns([3, 1])
        with col_lang:
            target_lang = st.selectbox(
                "Query Language",
                options=["auto", "en", "hi", "ta"],
                format_func=lambda x: {
                    "auto": "🌐 Auto-Detect",
                    "en": "🇬🇧 English",
                    "hi": "🇮🇳 Hindi (हिन्दी)",
                    "ta": "🇮🇳 Tamil (தமிழ்)",
                }.get(x, x),
            )

        # Input Mode Selector
        input_mode = st.radio("Input Mode", ["🎤 Voice Audio (Record / Upload)", "✍️ Text Query"], horizontal=True)

        audio_bytes = None
        text_query = ""

        if "Voice" in input_mode:
            st.markdown("### 🎙️ Speak your question")
            
            # Record from mic
            try:
                from audio_recorder_streamlit import audio_recorder
                recorded_audio = audio_recorder(
                    text="Click to record query audio",
                    recording_color="#e74c3c",
                    neutral_color="#3498db",
                    icon_name="microphone",
                    icon_size="2x",
                )
                if recorded_audio:
                    audio_bytes = recorded_audio
                    st.audio(audio_bytes, format="audio/wav")
            except Exception as e:
                st.warning(f"Microphone recorder component notice: {e}")

            # Audio File Uploader
            uploaded_file = st.file_uploader("Or upload an audio file (.wav, .mp3)", type=["wav", "mp3"])
            if uploaded_file is not None:
                audio_bytes = uploaded_file.read()
                st.audio(audio_bytes, format="audio/wav")

        else:
            # Preset sample pills
            st.markdown("**Sample Questions (Click to test):**")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🇬🇧 Capital of India?"):
                    text_query = "What is the capital of India?"
                if st.button("🇬🇧 What is RAG?"):
                    text_query = "What is Retrieval-Augmented Generation?"
            with col2:
                if st.button("🇮🇳 भारत की राजधानी?"):
                    text_query = "भारत की राजधानी क्या है?"
                if st.button("🇮🇳 गोवा कहाँ स्थित है?"):
                    text_query = "गोवा कहाँ स्थित है?"
            with col3:
                if st.button("🇮🇳 இந்தியாவின் தலைநகரம்?"):
                    text_query = "இந்தியாவின் தலைநகரம் எது?"
                if st.button("🇮🇳 உலகின் பெருங்கடல்?"):
                    text_query = "உலகின் மிகப்பெரிய பெருங்கடல் எது?"

            text_query = st.text_input("Enter your question:", value=text_query, placeholder="Ask anything in English, Hindi, or Tamil...")

        # Process Button
        submit_btn = st.button("🚀 Process Query", type="primary")

        if submit_btn:
            response: PipelineResponse = None
            with st.spinner("Processing through Voice Indic RAG pipeline..."):
                if "Voice" in input_mode and audio_bytes:
                    lang_code_map = {"en": "en-IN", "hi": "hi-IN", "ta": "ta-IN", "auto": "hi-IN"}
                    selected_code = lang_code_map.get(target_lang, "hi-IN")
                    response = orchestrator.process_audio(audio_bytes, language_code=selected_code)
                elif text_query:
                    selected_code = "en" if target_lang == "auto" else target_lang
                    response = orchestrator.process_query(text_query, language_code=selected_code)
                else:
                    st.warning("Please record audio, upload a file, or type a question.")

            if response:
                st.markdown("---")
                # Top Metrics
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-value">{response.latency.total_end_to_end_ms:.1f} ms</div>
                            <div class="metric-label">Total Latency (Budget: &lt;200ms)</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-value">{response.latency.total_retrieval_ms:.1f} ms</div>
                            <div class="metric-label">Hybrid Retrieval Latency</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with m3:
                    is_safe = response.input_guard.get("is_safe", True)
                    status_text = "PASSED" if is_safe else "BLOCKED"
                    status_color = "#2ecc71" if is_safe else "#e74c3c"
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-value" style="color: {status_color};">{status_text}</div>
                            <div class="metric-label">Input Guardrail</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with m4:
                    grounding_score = response.output_guard.get("grounding_score", 1.0) * 100
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-value">{grounding_score:.0f}%</div>
                            <div class="metric-label">Context Grounding Score</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                st.markdown("### 💬 Answer")
                if response.is_abstention:
                    st.markdown(
                        f"""<div class="abstention-box">
                            <strong>⚠️ Safe Abstention / Refusal:</strong><br/>
                            {response.answer}
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.success(response.answer)

                st.caption(f"**Query:** *{response.query}* | **Detected Language:** `{response.detected_lang}` | **Synthesizer:** `{response.provider}`")

                # Breakdown Section
                col_left, col_right = st.columns([1, 1])

                with col_left:
                    st.markdown("#### ⏱️ Latency Waterfall Breakdown")
                    breakdown_data = {
                        "STT (Speech-to-Text)": response.latency.stt_ms,
                        "Input Guardrail": response.latency.input_guardrail_ms,
                        "BGE-M3 Query Embedding": response.latency.embedding_ms,
                        "FAISS HNSW Dense Search": response.latency.dense_search_ms,
                        "BM25s Lexical Search": response.latency.lexical_search_ms,
                        "RRF Fusion": response.latency.fusion_ms,
                        "LLM Answer Generation": response.latency.llm_generation_ms,
                        "Output Guardrail Check": response.latency.output_guardrail_ms,
                    }
                    st.bar_chart(breakdown_data)

                with col_right:
                    st.markdown(f"#### 📖 Retrieved Context Passages ({len(response.retrieved_documents)})")
                    if not response.retrieved_documents:
                        st.info("No passages retrieved.")
                    for i, doc in enumerate(response.retrieved_documents):
                        with st.expander(f"[{i+1}] Doc ID: `{doc.passage_id}` (RRF: {doc.rrf_score:.4f}, Dense Cosine: {doc.dense_score:.3f})", expanded=(i == 0)):
                            st.markdown(f"**Text:** {doc.raw_text}")
                            st.caption(f"**Language:** `{doc.lang}` | **Chunk ID:** `{doc.chunk_id}`")

    with tab_benchmark:
        st.header("📈 Latency Benchmark & Statistical Analytics")
        if "benchmark_results" in st.session_state:
            b_res = st.session_state["benchmark_results"]
            summary = b_res["summary"]
            e2e = summary["total_end_to_end"]

            st.markdown(f"**Evaluated Queries:** {b_res['num_queries']} | **Languages:** `{', '.join(b_res['languages'])}`")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("P50 (Median) Latency", f"{e2e['p50']:.1f} ms")
            c2.metric("P70 Latency", f"{e2e['p70']:.1f} ms")
            c3.metric("P90 Latency", f"{e2e['p90']:.1f} ms")
            c4.metric("P100 (Max) Latency", f"{e2e['p100']:.1f} ms")

            st.subheader("Component Latency Percentiles (ms)")
            table_data = []
            for comp, s in summary.items():
                table_data.append({
                    "Component": comp,
                    "Min (ms)": f"{s['min']:.2f}",
                    "P50 (ms)": f"{s['p50']:.2f}",
                    "P70 (ms)": f"{s['p70']:.2f}",
                    "P90 (ms)": f"{s['p90']:.2f}",
                    "P100 (ms)": f"{s['p100']:.2f}",
                    "Mean (ms)": f"{s['mean']:.2f}",
                })
            st.dataframe(table_data, use_container_width=True)
        else:
            st.info("Click '⚡ Run Latency Benchmark' in the sidebar to generate live P50/P70/P100 latency analytics.")

    with tab_specs:
        st.header("📐 System Design & Technical Requirements")
        st.markdown(
            """
            ### 1. Speech-to-Text (STT)
            - **Provider**: Sarvam AI (`saaras:v2`) for Hindi, Tamil, and Indian English with code-mixing support.
            
            ### 2. Multi-Tiered Chunking Engine
            - **Strategies**: Atomic Passage, Sliding-Window with Overlap, Metadata-Augmented, and Query-Anchor pairing.
            
            ### 3. Sub-200ms Hybrid Retrieval
            - **Dense Index**: FAISS HNSW graph index on L2-normalized BAAI/bge-m3 embeddings (<10ms).
            - **Lexical Index**: BM25s with multilingual Devanagari & Tamil tokenization (<2ms).
            - **Fusion**: Reciprocal Rank Fusion (RRF, $k=60$) combining dense semantic and keyword signals.
            
            ### 4. Guardrails & Abstention
            - **Input Guard**: Prompt injection, toxicity, and acoustic confidence thresholding.
            - **Output Guard**: Strict context overlap grounding and automatic abstention on low relevance.
            """
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run_server":
        # Direct CLI helper
        pass
    else:
        # Check if launched via `python -m src.app`
        if not os.environ.get("STREAMLIT_SERVER_PORT"):
            os.environ["STREAMLIT_SERVER_PORT"] = "8501"
            os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
            cmd = ["streamlit", "run", __file__, "--server.port=8501", "--server.headless=true"]
            try:
                subprocess.run(cmd)
            except KeyboardInterrupt:
                pass
        else:
            main()
