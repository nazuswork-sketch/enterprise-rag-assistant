import os
import uuid
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.config import settings
from src.embeddings import embedding_client

class HybridVectorStore:
    """Qdrant Vector Store (Cloud or Local Embedded) with BM25 Sparse Hybrid Fusion."""

    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        storage_path: str = None,
        collection_name: str = None
    ):
        self.url = url if url is not None else settings.QDRANT_URL
        self.api_key = api_key if api_key is not None else settings.QDRANT_API_KEY
        self.storage_path = storage_path or settings.QDRANT_STORAGE_PATH
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        
        if self.url:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key if self.api_key else None,
                timeout=60
            )
        else:
            os.makedirs(self.storage_path, exist_ok=True)
            self.client = QdrantClient(path=self.storage_path)

        self._init_collection()
        
        self.bm25_corpus: List[Dict[str, Any]] = []
        self.bm25_index: BM25Okapi | None = None
        self._rebuild_bm25()

    def _init_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=models.Distance.COSINE
                )
            )

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in text.split() if len(w) > 1]

    def _rebuild_bm25(self):
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            self.bm25_corpus = [r.payload for r in records if r.payload]
            tokenized_corpus = [self._tokenize(doc.get("text", "")) for doc in self.bm25_corpus]
            if tokenized_corpus:
                self.bm25_index = BM25Okapi(tokenized_corpus)
            else:
                self.bm25_index = None
        except Exception as e:
            print(f"BM25 build note: {e}", flush=True)
            self.bm25_corpus = []
            self.bm25_index = None

    def add_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 50) -> int:
        if not chunks:
            return 0
            
        total_chunks = len(chunks)
        print(f"-> Starting batch embedding & cloud upsert for {total_chunks} chunks...", flush=True)
        
        total_points = 0
        for b_idx in range(0, total_chunks, batch_size):
            batch_chunks = chunks[b_idx:b_idx + batch_size]
            batch_texts = [c["text"] for c in batch_chunks]
            
            # Embed current batch
            batch_embeddings = embedding_client.embed_documents(batch_texts, batch_size=len(batch_texts))
            
            points = []
            for i, chunk in enumerate(batch_chunks):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.get("chunk_id", str(uuid.uuid4()))))
                points.append(models.PointStruct(
                    id=point_id,
                    vector=batch_embeddings[i],
                    payload=chunk
                ))
                
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            total_points += len(points)
            print(f"  [Qdrant Cloud] Indexed {total_points}/{total_chunks} chunks into '{self.collection_name}'", flush=True)
            
        # Update BM25 in memory directly
        self.bm25_corpus = [c for c in chunks]
        tokenized_corpus = [self._tokenize(doc.get("text", "")) for doc in self.bm25_corpus]
        if tokenized_corpus:
            self.bm25_index = BM25Okapi(tokenized_corpus)
            
        return total_points

    def search_dense(self, query: str, limit: int = 15) -> List[Dict[str, Any]]:
        query_vector = embedding_client.embed_query(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        ).points
        
        dense_docs = []
        for r in results:
            doc = dict(r.payload)
            doc["dense_score"] = float(r.score)
            dense_docs.append(doc)
        return dense_docs

    def search_sparse(self, query: str, limit: int = 15) -> List[Dict[str, Any]]:
        if not self.bm25_index or not self.bm25_corpus:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        
        scored_docs = []
        for doc, score in zip(self.bm25_corpus, scores):
            if score > 0:
                doc_copy = dict(doc)
                doc_copy["sparse_score"] = float(score)
                scored_docs.append(doc_copy)
                
        scored_docs.sort(key=lambda x: x["sparse_score"], reverse=True)
        return scored_docs[:limit]

    def hybrid_search(self, query: str, limit: int = settings.RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion (RRF) of Dense + Sparse Keyword Search."""
        dense_results = self.search_dense(query, limit=limit * 2)
        sparse_results = self.search_sparse(query, limit=limit * 2)
        
        k = 60
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        
        for rank, doc in enumerate(dense_results):
            cid = doc.get("chunk_id", doc.get("text"))
            doc_map[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
            
        for rank, doc in enumerate(sparse_results):
            cid = doc.get("chunk_id", doc.get("text"))
            if cid not in doc_map:
                doc_map[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
            
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        fused_docs = []
        for cid in sorted_cids[:limit]:
            doc = dict(doc_map[cid])
            doc["hybrid_score"] = round(rrf_scores[cid], 5)
            fused_docs.append(doc)
            
        return fused_docs

    def clear(self):
        self.client.delete_collection(self.collection_name)
        self._init_collection()
        self.bm25_corpus = []
        self.bm25_index = None

vector_store = HybridVectorStore()
