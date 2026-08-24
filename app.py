import os
import io
import sys
import json
import time
import pandas as pd
import streamlit as st
from pathlib import Path

# Ensure UTF-8 and project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import settings
from src.rag_engine import rag_engine
from src.parser import DocumentParser
from src.chunker import chunker
from src.vector_store import vector_store
from src.evaluation import RAGEvaluator
from src.observability import setup_observability, get_phoenix_url

# Page configuration
st.set_page_config(
    page_title="Enterprise Applied RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Start / check observability
phoenix_url = get_phoenix_url()

# Custom CSS for modern enterprise look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        background-color: #F1F5F9;
        color: #334155;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .source-card {
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        background: #F8FAFC;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.shields.io/badge/Architecture-Applied_RAG-blue?style=for-the-badge", use_container_width=True)
    st.markdown("### ⚙️ System Components")
    
    st.markdown(f"**🧠 Embedding Model:**  \n`{settings.GEMINI_EMBEDDING_MODEL}` (3072 dims)")
    st.markdown(f"**⚡ LLM Generator:**  \n`{settings.OPENROUTER_MODEL}`")
    st.markdown(f"**🎯 Cross-Encoder Reranker:**  \n`FlashRank (Local ONNX)`")
    st.markdown(f"**🗄️ Vector DB:**  \n`Embedded Qdrant (Zero Docker)`")
    st.markdown(f"**🔍 Sparse Index:**  \n`BM25 Okapi Hybrid Fusion`")
    
    st.divider()
    
    st.markdown("### 🔭 Live Observability")
    st.markdown("Arize Phoenix provides OpenInference tracing for every query.")
    if st.button("🚀 Launch Phoenix Tracing UI"):
        url = setup_observability()
        st.success(f"Running at [localhost:{settings.PHOENIX_PORT}]({url})")
        
    st.markdown(f"[🔗 Open Phoenix Dashboard (Port {settings.PHOENIX_PORT})]({phoenix_url})")

# ----------------- MAIN HEADER -----------------
st.markdown('<div class="main-header">🛡️ Enterprise Applied RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Heterogeneous Data Ingestion • Hybrid Search + Reranking • Hallucination Guards • Golden CI/CD Benchmarking</div>', unsafe_allow_html=True)

tab_chat, tab_docs, tab_eval, tab_arch = st.tabs([
    "💬 Knowledge Assistant",
    "📁 Document Ingestion",
    "📊 Golden CI/CD Benchmark",
    "🏗️ Pipeline Architecture"
])

# ----------------- TAB 1: CHAT ASSISTANT -----------------
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your Enterprise AI Assistant grounded in internal documentation (PDFs, Slack chats, runbooks). Ask me anything!"}
        ]

    # Sample query quick buttons
    st.markdown("**💡 Quick Sample Queries:**")
    col1, col2, col3 = st.columns(3)
    sample_q = None
    if col1.button("SEV-1 SLA & Kubernetes Rollback"):
        sample_q = "What is the response SLA for SEV-1 incidents and how do we roll back a kubernetes deployment?"
    if col2.button("Gemini Embedding Dimension & Qdrant"):
        sample_q = "What is the vector dimension for our new Gemini Embedding 2 model and how does it work with Qdrant?"
    if col3.button("L&D Stipend & AI API Credits"):
        sample_q = "How much is the annual learning and development stipend and can it be used for AI API credits?"

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander(f"📚 View {len(msg['sources'])} Grounded Sources & Retrieval Scores"):
                    for idx, s in enumerate(msg["sources"]):
                        st.markdown(f"""
                        **[{idx+1}] Source:** `{s['source']}` | **Rerank Score:** `{s['rerank_score']}` | **Hybrid Score:** `{s['hybrid_score']}`  
                        *Chunk ID:* `{s['chunk_id']}`  
                        > {s['full_text']}
                        """)
            if "telemetry" in msg:
                t = msg["telemetry"]
                st.caption(f"⚡ Total Latency: **{t.get('total_latency')}s** | 🔍 Retrieval: **{t.get('retrieval_latency')}s** | 🎯 Rerank: **{t.get('rerank_latency')}s** | 🤖 LLM: **{t.get('llm_latency')}s** | 🪙 Tokens: **{t.get('tokens')}**")

    # Handle user query
    user_input = st.chat_input("Ask a question about internal enterprise policies or infrastructure...")
    prompt = sample_q or user_input

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving hybrid contexts & reranking with FlashRank..."):
                result = rag_engine.query(prompt)
                
            st.markdown(result["answer"])
            
            if result["sources"]:
                with st.expander(f"📚 View {len(result['sources'])} Grounded Sources & Retrieval Scores"):
                    for idx, s in enumerate(result["sources"]):
                        st.markdown(f"""
                        **[{idx+1}] Source:** `{s['source']}` | **Rerank Score:** `{s['rerank_score']}` | **Hybrid Score:** `{s['hybrid_score']}`  
                        *Chunk ID:* `{s['chunk_id']}`  
                        > {s['full_text']}
                        """)
                        
            telemetry = {
                "total_latency": result["total_latency_seconds"],
                "retrieval_latency": result["retrieval_metadata"]["retrieval_latency"],
                "rerank_latency": result["retrieval_metadata"]["rerank_latency"],
                "llm_latency": result["generation_metadata"]["latency_seconds"],
                "tokens": result["generation_metadata"]["total_tokens"]
            }
            
            st.caption(f"⚡ Total Latency: **{telemetry['total_latency']}s** | 🔍 Retrieval: **{telemetry['retrieval_latency']}s** | 🎯 Rerank: **{telemetry['rerank_latency']}s** | 🤖 LLM: **{telemetry['llm_latency']}s** | 🪙 Tokens: **{telemetry['tokens']}**")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "telemetry": telemetry
            })

# ----------------- TAB 2: INGESTION & KNOWLEDGE BASE -----------------
with tab_docs:
    st.markdown("### 📁 Ingestion & Knowledge Base Management")
    st.markdown("Upload documents (PDF, Markdown, Slack JSON exports, TXT) to automatically chunk and embed them into the embedded Qdrant store.")
    
    col_upload, col_action = st.columns([3, 1])
    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload Enterprise Documents",
            type=["pdf", "md", "txt", "json"],
            accept_multiple_files=True
        )
        
    with col_action:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📥 Save & Re-index Data", use_container_width=True):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    save_path = settings.DATA_DIR / uploaded_file.name
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success(f"Saved {len(uploaded_files)} files to {settings.DATA_DIR}")
                
            with st.spinner("Parsing, chunking, and embedding vectors..."):
                raw_docs = DocumentParser.parse_directory(settings.DATA_DIR)
                chunks = chunker.chunk_documents(raw_docs)
                vector_store.clear()
                count = vector_store.add_documents(chunks)
                st.success(f" Successfully indexed {count} chunks into Qdrant!")

    st.divider()
    st.markdown("### 🗂️ Active Documents in `data/` Directory")
    doc_files = list(settings.DATA_DIR.glob("*.*"))
    if doc_files:
        data_table = []
        for df in doc_files:
            data_table.append({
                "Filename": df.name,
                "Size (KB)": round(df.stat().st_size / 1024, 2),
                "Format": df.suffix.upper(),
                "Last Modified": time.ctime(df.stat().st_mtime)
            })
        st.dataframe(pd.DataFrame(data_table), use_container_width=True)
    else:
        st.info("No files currently in data directory.")

# ----------------- TAB 3: EVALUATION & GOLDEN DATASET -----------------
with tab_eval:
    st.markdown("### 📊 Automated Golden Dataset Evaluation & CI/CD Regression")
    st.markdown("Evaluates retrieval precision, context recall, and faithfulness (hallucination detection) using LLM-as-a-Judge against standard benchmarks.")
    
    available_datasets = []
    for d_name in ["synthetic_golden_qa.json", "golden_dataset.json"]:
        if (settings.BASE_DIR / d_name).exists():
            available_datasets.append(d_name)
            
    col_e1, col_e2 = st.columns([3, 1])
    with col_e1:
        selected_dataset = st.selectbox("Select Evaluation Dataset:", available_datasets, index=0)
        evaluator = RAGEvaluator(selected_dataset)
        st.caption(f"Loaded **{len(evaluator.load_dataset())}** Q&A Pairs from `{selected_dataset}`")
    with col_e2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_eval_clicked = st.button("🚀 Run Full Benchmark", type="primary", use_container_width=True)

    if run_eval_clicked:
        with st.spinner("Running RAG evaluation across all golden test cases..."):
            benchmark_results = evaluator.run_benchmark()
            st.session_state.benchmark_results = benchmark_results
            st.success(" Benchmark complete!")

    if "benchmark_results" in st.session_state:
        res = st.session_state.benchmark_results
        
        # Summary Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🛡️ Faithfulness", f"{res['avg_faithfulness']:.2f} / 1.00", help="1.0 = Zero Hallucinations")
        m2.metric("🎯 Context Precision", f"{res['avg_context_precision']:.2f} / 1.00", help="Accuracy of retrieved passages")
        m3.metric("📖 Context Recall", f"{res['avg_context_recall']:.2f} / 1.00", help="Coverage of ground truth facts")
        m4.metric("💬 Answer Relevancy", f"{res['avg_answer_relevancy']:.2f} / 1.00", help="Directness of answer")
        m5.metric("⚡ Avg Latency", f"{res['avg_latency_seconds']:.2f}s", help="End-to-end response time")
        
        st.markdown("#### 📋 Detailed Test Case Breakdown")
        detail_df = pd.DataFrame(res["detailed_results"])
        st.dataframe(detail_df[["question", "faithfulness", "context_precision", "context_recall", "answer_relevancy", "latency_seconds"]], use_container_width=True)
        
        with st.expander("🔍 View Full Evaluator Reasoning per Test Case"):
            for idx, r in enumerate(res["detailed_results"]):
                st.markdown(f"**Q{idx+1}: {r['question']}**")
                st.markdown(f"*Ground Truth:* {r['ground_truth']}")
                st.markdown(f"*Generated:* {r['generated_answer']}")
                st.markdown(f"*Judge Reasoning:* {r['reasoning']}")
                st.divider()

# ----------------- TAB 4: ARCHITECTURE -----------------
with tab_arch:
    st.markdown("### 🏗️ Enterprise Applied RAG Architecture")
    st.markdown("""
    #### 1. Ingestion & Intelligent Chunking
    * **Multi-Format Parsers**: PDF (`pypdf`), Markdown, Slack exports (`JSON`), Text.
    * **Recursive Semantic Chunker**: Splits along paragraph & sentence boundaries while attaching chunk IDs, source document names, and line/page numbers.

    #### 2. Zero-Docker Hybrid Vector Store
    * **Dense Vectors**: Google AI Studio **`gemini-embedding-2`** producing 3072-dimensional embeddings.
    * **Sparse Vectors**: In-memory **`BM25Okapi`** keyword index for exact keyword matching (acronyms, command names, error codes).
    * **Reciprocal Rank Fusion (RRF)**: Combines dense cosine similarity with sparse BM25 scores.

    #### 3. Cross-Encoder Reranker
    * **FlashRank**: Runs locally via ONNX (`ms-marco-TinyBERT-L-2-v2`). Re-ranks Top-K hybrid candidates to Top-N most relevant passages, filtering out context clutter.

    #### 4. Generation & Grounding
    * **LLM**: OpenRouter **`nvidia/nemotron-3-ultra-550b-a55b:free`**.
    * **Grounding & Citation Directives**: Strict system prompt enforcing factual citations and explicit fallback if documentation lacks context.

    #### 5. Evaluation & Observability
    * **CI/CD Evaluator (`run_eval.py`)**: Runs against a Golden Q&A dataset to measure **Faithfulness**, **Context Precision**, and **Recall**.
    * **Arize Phoenix**: Zero-Docker OpenInference tracing dashboard for end-to-end query visibility.
    """)
