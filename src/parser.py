import os
import json
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

class DocumentParser:
    """Parses heterogeneous documents into normalized text with metadata."""

    @staticmethod
    def parse_pdf(file_path: Path) -> List[Dict[str, Any]]:
        docs = []
        reader = PdfReader(str(file_path))
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append({
                    "text": text.strip(),
                    "source": file_path.name,
                    "doc_type": "pdf",
                    "page": page_idx + 1,
                    "file_path": str(file_path)
                })
        return docs

    @staticmethod
    def parse_markdown(file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        
        # Split by markdown headers (# or ##) if long, or return document
        sections = text.split("\n# ")
        docs = []
        if len(sections) > 1:
            for idx, sec in enumerate(sections):
                prefix = "" if idx == 0 else "# "
                sec_text = (prefix + sec).strip()
                if sec_text:
                    docs.append({
                        "text": sec_text,
                        "source": file_path.name,
                        "doc_type": "markdown",
                        "section_id": idx,
                        "file_path": str(file_path)
                    })
        else:
            docs.append({
                "text": text.strip(),
                "source": file_path.name,
                "doc_type": "markdown",
                "file_path": str(file_path)
            })
        return docs

    @staticmethod
    def parse_slack_json(file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        
        docs = []
        # Support array of messages or Slack export object
        messages = data if isinstance(data, list) else data.get("messages", [])
        channel = data.get("channel", "general") if isinstance(data, dict) else "general"
        
        for idx, msg in enumerate(messages):
            user = msg.get("user", msg.get("username", "Unknown"))
            text = msg.get("text", "")
            ts = msg.get("ts", "")
            thread_replies = msg.get("replies", [])
            
            formatted_text = f"Slack Message in #{channel} from @{user} (ts: {ts}):\n{text}"
            if thread_replies:
                formatted_text += "\nReplies in thread: " + " | ".join([r.get("text", "") for r in thread_replies if "text" in r])
                
            if text.strip():
                docs.append({
                    "text": formatted_text.strip(),
                    "source": file_path.name,
                    "doc_type": "slack",
                    "channel": channel,
                    "author": user,
                    "message_idx": idx,
                    "file_path": str(file_path)
                })
        return docs

    @staticmethod
    def parse_text(file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [{
            "text": text.strip(),
            "source": file_path.name,
            "doc_type": "text",
            "file_path": str(file_path)
        }]

    @classmethod
    def parse_file(cls, file_path: str | Path) -> List[Dict[str, Any]]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return cls.parse_pdf(path)
        elif suffix in [".md", ".markdown"]:
            return cls.parse_markdown(path)
        elif suffix == ".json":
            return cls.parse_slack_json(path)
        elif suffix in [".txt", ".log", ".csv"]:
            return cls.parse_text(path)
        else:
            return cls.parse_text(path)

    @classmethod
    def parse_directory(cls, dir_path: str | Path) -> List[Dict[str, Any]]:
        all_docs = []
        path = Path(dir_path)
        if not path.exists():
            return all_docs
            
        for file in path.glob("**/*"):
            if file.is_file() and not file.name.startswith("."):
                try:
                    docs = cls.parse_file(file)
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"Warning: Failed to parse {file}: {e}")
        return all_docs
