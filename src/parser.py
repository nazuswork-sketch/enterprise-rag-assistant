import os
import json
import base64
import io
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from PIL import Image

class DocumentParser:
    """Parses heterogeneous documents into normalized text and multimodal image assets with metadata."""

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
                
            # Multimodal: Extract high-value diagrams/figures from PDF page
            if hasattr(page, "images"):
                for img_idx, img in enumerate(page.images):
                    try:
                        img_bytes = img.data
                        # Verify image validity and dimension
                        pil_img = Image.open(io.BytesIO(img_bytes))
                        if pil_img.width >= 60 and pil_img.height >= 60:
                            # Optimize image size for fast cloud transfer
                            if pil_img.width > 800 or pil_img.height > 800:
                                pil_img.thumbnail((800, 800))
                                out_buf = io.BytesIO()
                                pil_img.convert("RGB").save(out_buf, format="JPEG", quality=85)
                                img_bytes = out_buf.getvalue()
                                mime_type = "image/jpeg"
                            else:
                                mime_type = "image/png"
                                
                            b64_str = base64.b64encode(img_bytes).decode("utf-8")
                            docs.append({
                                "text": f"Visual Diagram/Drawing on Page {page_idx+1} of {file_path.name}: {img.name}",
                                "source": file_path.name,
                                "doc_type": "image",
                                "page": page_idx + 1,
                                "image_name": f"{file_path.stem}_p{page_idx+1}_fig{img_idx+1}",
                                "image_base64": b64_str,
                                "mime_type": mime_type,
                                "file_path": str(file_path)
                            })
                    except Exception as e:
                        pass
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

    @staticmethod
    def parse_image(file_path: Path) -> List[Dict[str, Any]]:
        """Parses standalone image assets (diagrams, drawings, charts)."""
        try:
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            pil_img = Image.open(io.BytesIO(img_bytes))
            
            # Compress / optimize for fast multimodal embedding
            if pil_img.width > 800 or pil_img.height > 800:
                pil_img.thumbnail((800, 800))
                out_buf = io.BytesIO()
                pil_img.convert("RGB").save(out_buf, format="JPEG", quality=85)
                img_bytes = out_buf.getvalue()
                mime_type = "image/jpeg"
            else:
                ext = file_path.suffix.lstrip(".").lower()
                mime_type = f"image/{ext if ext != 'jpg' else 'jpeg'}"
                
            b64_str = base64.b64encode(img_bytes).decode("utf-8")
            return [{
                "text": f"Visual Architecture/Engineering Image: {file_path.name}",
                "source": file_path.name,
                "doc_type": "image",
                "image_name": file_path.stem,
                "image_base64": b64_str,
                "mime_type": mime_type,
                "file_path": str(file_path)
            }]
        except Exception as e:
            print(f"Warning: Failed to parse image {file_path}: {e}")
            return []

    @classmethod
    def parse_file(cls, file_path: str | Path) -> List[Dict[str, Any]]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return cls.parse_pdf(path)
        elif suffix in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"]:
            return cls.parse_image(path)
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
