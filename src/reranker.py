from typing import List, Dict, Any
from flashrank import Ranker, RerankRequest
from src.config import settings

class LocalFlashRanker:
    """Ultra-fast, zero-Docker, local cross-encoder reranker."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        # FlashRank downloads a tiny ~15MB ONNX model once locally
        self.ranker = Ranker(model_name=model_name, cache_dir=str(settings.STORAGE_DIR / "flashrank_cache"))

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = settings.RERANK_TOP_N) -> List[Dict[str, Any]]:
        if not documents:
            return []
            
        passages = [
            {"id": i, "text": doc.get("text", ""), "meta": doc}
            for i, doc in enumerate(documents)
        ]
        
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)
        
        reranked_docs = []
        for item in results[:top_n]:
            meta = dict(item.get("meta", {}))
            meta["rerank_score"] = round(float(item.get("score", 0.0)), 4)
            reranked_docs.append(meta)
            
        return reranked_docs

reranker = LocalFlashRanker()
