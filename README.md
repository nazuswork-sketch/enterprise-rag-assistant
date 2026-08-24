# Enterprise Applied RAG with Strict Evaluation & Monitoring

An enterprise-ready Retrieval-Augmented Generation (RAG) system built to answer employee questions over heterogeneous internal documentation (PDFs, Markdown runbooks, and Slack exports) with **zero hallucinations**, **automated CI/CD evaluation**, and **full-trace observability** -- running **100% Python with Zero Docker**.

---

## Architecture & Tech Stack

- **Embeddings:** Google AI Studio **`gemini-embedding-2`** (3072 dimensions).
- **Generation:** NVIDIA Nemotron via OpenRouter (**`nvidia/nemotron-3-ultra-550b-a55b:free`**).
- **Vector Database (Zero-Docker):** **Embedded Qdrant** (`./storage/qdrant_db`) paired with **BM25 Okapi** for Hybrid Fusion Search.
- **Reranker:** **FlashRank** local ONNX cross-encoder (`ms-marco-TinyBERT-L-2-v2`).
- **Evaluation & CI/CD:** Automated Golden Dataset LLM-as-a-Judge benchmark measuring **Faithfulness**, **Context Precision**, and **Recall**.
- **Observability:** **Arize Phoenix** (Zero-Docker OpenInference tracing UI on `http://localhost:6006`).
- **User Interface:** **Streamlit** multi-tab interactive knowledge assistant & metrics dashboard.

---

## Project Structure

```
project1/
├── data/                             # Raw enterprise docs (PDF, Markdown, Slack JSON, TXT)
│   ├── devops_incident_runbook.md
│   ├── engineering_slack_chat.json
│   └── corporate_security_policy.txt
├── storage/                          # Local embedded Qdrant database (Zero Docker)
├── src/
│   ├── config.py                     # Environment & hyperparameters
│   ├── embeddings.py                 # Gemini Embedding 2 client (3072-dim)
│   ├── llm.py                        # OpenRouter NVIDIA Nemotron client
│   ├── parser.py                     # Multi-format document parser
│   ├── chunker.py                    # Intelligent recursive semantic chunker
│   ├── vector_store.py               # Embedded Qdrant + BM25 Hybrid Store
│   ├── reranker.py                   # FlashRank cross-encoder reranker
│   ├── rag_engine.py                 # Core RAG pipeline with citation grounding
│   ├── observability.py              # Arize Phoenix tracing integration
│   └── evaluation.py                 # Benchmark engine for golden dataset
├── golden_dataset.json               # Curated Q&A test cases with ground truths
├── ingest.py                         # Ingestion pipeline runner
├── run_eval.py                       # CI/CD evaluation benchmark runner
├── app.py                            # Streamlit chat & analytics dashboard
├── requirements.txt                  # Python dependencies
└── .env                              # API keys & configuration
```

---

## How to Run

### 1. Ingest Documents into Local Qdrant
```bash
python ingest.py
```

### 2. Run the Streamlit Web Application
```bash
streamlit run app.py
```
* **Tab 1 (Chat):** Query the knowledge base with real-time citations, expandable source cards, and latency/token breakdown.
* **Tab 2 (Ingestion):** Upload new PDFs, Slack logs, or Markdown docs and re-index.
* **Tab 3 (Evaluation):** Run the Golden Dataset evaluation benchmark on demand.
* **Tab 4 (Architecture):** Interactive diagram & component breakdown.

### 3. Run CI/CD Regression Evaluation
```bash
python run_eval.py
```
Outputs a summary table and writes full metric logs to `eval_report.json`.

### 4. Open Observability Dashboard
When enabled, open `http://localhost:6006` to inspect full OpenInference execution traces, span trees, and token consumption breakdowns.
