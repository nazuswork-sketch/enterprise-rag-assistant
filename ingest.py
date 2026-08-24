import sys
from pathlib import Path
from src.config import settings
from src.parser import DocumentParser
from src.chunker import chunker
from src.vector_store import vector_store

def run_ingestion(data_dir: Path = settings.DATA_DIR):
    print(f"Starting Ingestion from: {data_dir}")
    
    # 1. Parse documents
    raw_docs = DocumentParser.parse_directory(data_dir)
    print(f"-> Parsed {len(raw_docs)} document sections.")
    
    if not raw_docs:
        print("No documents found to ingest.")
        return 0
        
    # 2. Chunk documents
    chunks = chunker.chunk_documents(raw_docs)
    print(f"-> Generated {len(chunks)} intelligent chunks.")
    
    # 3. Add to Qdrant Cloud + BM25 index
    print(f"-> Indexing chunks into Qdrant Vector Store ({'Cloud' if settings.QDRANT_URL else 'Local'}) with Gemini Embeddings...")
    vector_store.clear()
    count = vector_store.add_documents(chunks)
    print(f" SUCCESS! Ingested {count} vectors into '{settings.QDRANT_COLLECTION_NAME}'")
    return count

if __name__ == "__main__":
    run_ingestion()
