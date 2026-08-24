import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / '.env', override=True)

class Settings:
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / 'data'
    STORAGE_DIR: Path = BASE_DIR / 'storage'
    
    # Google AI Studio Embeddings
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    GEMINI_EMBEDDING_MODEL: str = os.getenv('GEMINI_EMBEDDING_MODEL', 'models/gemini-embedding-2')
    EMBEDDING_DIM: int = 3072
    
    # OpenRouter LLM (NVIDIA Nemotron)
    OPENROUTER_API_KEY: str = os.getenv('OPENROUTER_API_KEY', '')
    OPENROUTER_MODEL: str = os.getenv('OPENROUTER_MODEL', 'nvidia/nemotron-3-ultra-550b-a55b:free')
    OPENROUTER_BASE_URL: str = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
    
    # Qdrant Vector DB (Cloud or Local Embedded)
    QDRANT_URL: str = os.getenv('QDRANT_URL', '')
    QDRANT_API_KEY: str = os.getenv('QDRANT_API_KEY', '')
    QDRANT_STORAGE_PATH: str = os.getenv('QDRANT_STORAGE_PATH', str(STORAGE_DIR / 'qdrant_db'))
    QDRANT_COLLECTION_NAME: str = os.getenv('QDRANT_COLLECTION_NAME', 'enterprise_knowledge_base')
    
    # RAG Retrieval Configuration
    RETRIEVAL_TOP_K: int = 15     # Number of candidates from hybrid search
    RERANK_TOP_N: int = 5         # Number of candidates sent to LLM after reranker
    CHUNK_SIZE: int = 1500        # Optimized semantic chunk size for high-precision fact lookup
    CHUNK_OVERLAP: int = 200
    
    # Observability
    PHOENIX_PORT: int = 6006

settings = Settings()
