import json
import time
import urllib.request
import urllib.error
from typing import List
from src.config import settings

class GeminiEmbeddingClient:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.models = [
            "models/gemini-embedding-001",
            "models/gemini-embedding-2",
            "models/gemini-embedding-2-preview"
        ]
        self.active_model = model or "models/gemini-embedding-001"
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

    def _get_urls(self, model: str):
        if not model.startswith("models/"):
            model = f"models/{model}"
        single_url = f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent?key={self.api_key}"
        batch_url = f"https://generativelanguage.googleapis.com/v1beta/{model}:batchEmbedContents?key={self.api_key}"
        return single_url, batch_url

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text string with multi-model fallback."""
        if not text or not text.strip():
            return [0.0] * settings.EMBEDDING_DIM
        
        for model in self.models:
            single_url, _ = self._get_urls(model)
            payload = {
                "model": model,
                "content": {"parts": [{"text": text.strip()}]}
            }
            req = urllib.request.Request(
                single_url,
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    res = json.loads(response.read().decode('utf-8'))
                    values = res.get("embedding", {}).get("values", [])
                    if values:
                        return values
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    continue  # Try next available model in quota pool
            except Exception:
                continue
                
        return [0.0] * settings.EMBEDDING_DIM

    def embed_documents(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Embed document chunks in batches with automatic model fallback and retries."""
        all_embeddings = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_success = False
            
            # Retry loop for batch (up to 5 retries with backoff)
            for attempt in range(5):
                for model in self.models:
                    _, batch_url = self._get_urls(model)
                    requests_list = [
                        {
                            "model": model,
                            "content": {"parts": [{"text": t.strip() if t.strip() else " "}]}
                        }
                        for t in batch_texts
                    ]
                    payload = {"requests": requests_list}
                    req = urllib.request.Request(
                        batch_url,
                        data=json.dumps(payload).encode('utf-8'),
                        headers={"Content-Type": "application/json"}
                    )
                    
                    try:
                        with urllib.request.urlopen(req, timeout=90) as response:
                            res = json.loads(response.read().decode('utf-8'))
                            embs = res.get("embeddings", [])
                            if embs and len(embs) == len(batch_texts):
                                for item in embs:
                                    all_embeddings.append(item.get("values", [0.0] * settings.EMBEDDING_DIM))
                                batch_success = True
                                break
                    except urllib.error.HTTPError as e:
                        if e.code == 429:
                            time.sleep(2.0 * (attempt + 1))
                            continue
                    except Exception as e:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                        
                if batch_success:
                    break
                    
            if not batch_success:
                print(f"  [Embeddings Warning] Rate limit persists on batch {i}-{i+batch_size}, padding fallback.", flush=True)
                for _ in batch_texts:
                    all_embeddings.append([0.0] * settings.EMBEDDING_DIM)
                    
            print(f"  [Embeddings] Processed {len(all_embeddings)}/{total} chunks...", flush=True)
            time.sleep(1.0)
            
        return all_embeddings

embedding_client = GeminiEmbeddingClient()
