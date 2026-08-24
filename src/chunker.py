import re
from typing import List, Dict, Any
from src.config import settings

class IntelligentChunker:
    """Recursively splits text on semantic boundaries while retaining metadata."""
    
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
                
            # Find best separator before end
            sub = text[start:end]
            split_idx = -1
            for sep in self.separators:
                last_pos = sub.rfind(sep)
                if last_pos != -1 and last_pos > self.chunk_size // 3:
                    split_idx = last_pos + len(sep)
                    break
                    
            if split_idx == -1:
                split_idx = self.chunk_size
                
            chunk = text[start:start + split_idx].strip()
            if chunk:
                chunks.append(chunk)
                
            start += max(split_idx - self.chunk_overlap, 1)
            
        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunked_docs = []
        global_chunk_idx = 0
        
        for doc in documents:
            # Preserve visual diagrams / image assets intact
            if doc.get("doc_type") == "image":
                img_meta = {**doc}
                img_meta["chunk_id"] = f"{doc.get('source', 'img')}_img_{global_chunk_idx}"
                img_meta["chunk_index"] = 0
                img_meta["total_chunks_in_doc"] = 1
                chunked_docs.append(img_meta)
                global_chunk_idx += 1
                continue
                
            text = doc.get("text", "")
            raw_chunks = self._split_text(text)
            
            for local_idx, chunk in enumerate(raw_chunks):
                chunk_meta = {**doc}
                chunk_meta["chunk_id"] = f"{doc.get('source', 'doc')}_{local_idx}_{global_chunk_idx}"
                chunk_meta["chunk_index"] = local_idx
                chunk_meta["total_chunks_in_doc"] = len(raw_chunks)
                chunk_meta["text"] = chunk
                chunked_docs.append(chunk_meta)
                global_chunk_idx += 1
                
        return chunked_docs

chunker = IntelligentChunker()
