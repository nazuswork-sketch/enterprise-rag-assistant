from typing import List, Dict, Any
from flashrank import Ranker, RerankRequest
from src.config import settings

class LocalFlashRanker:
    """Ultra-fast, zero-Docker, local cross-encoder reranker."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self._ranker = None

    @property
    def ranker(self) -> Ranker:
        if self._ranker is None:
            try:
                self._ranker = Ranker(model_name=self.model_name, cache_dir=str(settings.STORAGE_DIR / "flashrank_cache"))
            except Exception as e:
                print(f"[Reranker Init Note] {e}")
        return self._ranker

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = settings.RERANK_TOP_N) -> List[Dict[str, Any]]:
        if not documents:
            return []
            
        passages = [
            {"id": i, "text": doc.get("text", ""), "meta": doc}
            for i, doc in enumerate(documents)
        ]
        
        try:
            if self.ranker:
                rerank_request = RerankRequest(query=query, passages=passages)
                results = self.ranker.rerank(rerank_request)
                
                reranked_docs = []
                for item in results[:top_n]:
                    meta = dict(item.get("meta", {}))
                    meta["rerank_score"] = round(float(item.get("score", 0.0)), 4)
                    reranked_docs.append(meta)
                if reranked_docs:
                    return reranked_docs
        except Exception as e:
            print(f"[Reranker Warning] {e}")
            
        return documents[:top_n]

reranker = LocalFlashRanker()
