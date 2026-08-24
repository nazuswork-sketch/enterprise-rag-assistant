import time
from typing import Dict, Any, List, Generator
from src.config import settings
from src.vector_store import vector_store
from src.reranker import reranker
from src.llm import llm_client

SYSTEM_PROMPT = """You are an Enterprise AI Knowledge Assistant.
Answer the user's question accurately and concisely based ONLY on the provided context passages.

RULES:
1. Ground every claim directly in the context. DO NOT hallucinate or assume facts not present in the context.
2. If the context does not contain enough information to answer the question, clearly state: "Based on the provided internal documentation, I do not have enough information to answer this question."
3. Cite your sources inline using [Source: document_name, chunk_id].
4. Format your response cleanly using markdown with bullet points and sections where appropriate.
"""

class EnterpriseRAGEngine:
    """Complete Enterprise RAG Pipeline uniting Hybrid Search, FlashRank Reranker, and Nemotron LLM."""

    def __init__(self):
        self.vector_store = vector_store
        self.reranker = reranker
        self.llm = llm_client

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Perform Hybrid Search + Reranking."""
        t0 = time.time()
        hybrid_candidates = self.vector_store.hybrid_search(query, limit=settings.RETRIEVAL_TOP_K)
        retrieval_latency = time.time() - t0
        
        t1 = time.time()
        reranked_docs = self.reranker.rerank(query, hybrid_candidates, top_n=settings.RERANK_TOP_N)
        rerank_latency = time.time() - t1
        
        return {
            "query": query,
            "raw_candidates": hybrid_candidates,
            "reranked_docs": reranked_docs,
            "retrieval_latency": round(retrieval_latency, 3),
            "rerank_latency": round(rerank_latency, 3)
        }

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        context_blocks = []
        for i, doc in enumerate(docs):
            src = doc.get("source", "doc")
            cid = doc.get("chunk_id", f"chunk_{i}")
            doc_type = doc.get("doc_type", "general")
            text = doc.get("text", "")
            context_blocks.append(f"--- [PASSAGE {i+1} | Source: {src} | Type: {doc_type} | ID: {cid}] ---\n{text}")
        return "\n\n".join(context_blocks)

    def query(self, query_text: str) -> Dict[str, Any]:
        """Execute full RAG pipeline and return response with metadata."""
        retrieval_data = self.retrieve(query_text)
        retrieved_docs = retrieval_data["reranked_docs"]
        
        if not retrieved_docs:
            return {
                "answer": "No relevant documents found in the enterprise knowledge base.",
                "sources": [],
                "retrieval_metadata": retrieval_data,
                "generation_metadata": {"latency_seconds": 0.0, "total_tokens": 0},
                "total_latency_seconds": round(retrieval_data["retrieval_latency"] + retrieval_data["rerank_latency"], 3)
            }
            
        context_str = self._format_context(retrieved_docs)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context passages:\n{context_str}\n\nQuestion: {query_text}"}
        ]
        
        gen_result = self.llm.generate(messages)
        total_latency = retrieval_data["retrieval_latency"] + retrieval_data["rerank_latency"] + gen_result["latency_seconds"]
        has_error = bool(gen_result.get("error"))
        
        return {
            "answer": gen_result["content"],
            "is_error": has_error,
            "error_detail": gen_result.get("error"),
            "sources": [] if has_error else [
                {
                    "source": d.get("source"),
                    "doc_type": d.get("doc_type"),
                    "chunk_id": d.get("chunk_id"),
                    "rerank_score": d.get("rerank_score"),
                    "hybrid_score": d.get("hybrid_score"),
                    "text_preview": d.get("text", "")[:200] + "...",
                    "full_text": d.get("text", "")
                }
                for d in retrieved_docs
            ],
            "retrieval_metadata": retrieval_data,
            "generation_metadata": gen_result,
            "total_latency_seconds": round(total_latency, 3)
        }

rag_engine = EnterpriseRAGEngine()
