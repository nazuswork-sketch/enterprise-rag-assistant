import os
import re
import sys
import json
import time
import argparse
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import settings
from src.parser import DocumentParser
from src.llm import llm_client

PROMPT_TEMPLATE = """Based on the following text, generate 3 complex, multi-hop questions that require reading multiple paragraphs to answer. Also provide the exact ground-truth answer based ONLY on the text. Output as JSON.

Format your response strictly as a JSON array of objects:
[
  {
    "question": "Complex multi-hop question here...",
    "ground_truth": "Exact fact-based answer derived strictly from the text..."
  }
]

TEXT TO ANALYZE:
"""

def extract_json(content: str):
    """Extract and parse JSON array from model output using markdown stripping and regex."""
    content = content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
        
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "questions" in data:
            return data["questions"]
        return [data]
    except Exception:
        # Fallback: search with regex for JSON array
        match = re.search(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return []

def generate_qa_from_pdfs(
    data_dir: Path = settings.DATA_DIR,
    output_file: str = "synthetic_golden_qa.json",
    num_sections_per_pdf: int = 2,
    chunk_char_size: int = 3500
):
    print("=" * 70)
    print("🚀 GENERATING MULTI-HOP Q&A DATASET FROM PDF DOCUMENTS")
    print(f"🤖 Model: {settings.OPENROUTER_MODEL}")
    print(f"📁 Source Directory: {data_dir}")
    print("=" * 70)

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found.")
        return []

    all_generated_qa = []
    global_id = 1

    for pdf_path in pdf_files:
        print(f"\n📄 [FILE] Processing: {pdf_path.name}")
        pages = DocumentParser.parse_pdf(pdf_path)
        print(f"   Extracted {len(pages)} pages.")

        combined_sections = []
        current_section = ""

        for p in pages:
            text = p.get("text", "")
            if len(text.split()) < 60:
                continue
            current_section += f"\n\n[Page {p.get('page')}]\n" + text
            if len(current_section) >= chunk_char_size:
                combined_sections.append(current_section.strip())
                current_section = ""

        if current_section:
            combined_sections.append(current_section.strip())

        step = max(1, len(combined_sections) // num_sections_per_pdf)
        selected_sections = [combined_sections[i] for i in range(0, len(combined_sections), step)][:num_sections_per_pdf]

        print(f"   Feeding {len(selected_sections)} multi-paragraph sections to OpenRouter...")

        for s_idx, section_text in enumerate(selected_sections):
            prompt = PROMPT_TEMPLATE + section_text[:4000] + '\n"""\n'
            messages = [
                {"role": "system", "content": "You are an expert AI evaluator. Analyze the text carefully, follow all instructions, and output only valid JSON."},
                {"role": "user", "content": prompt}
            ]

            t0 = time.time()
            response = llm_client.generate(messages, temperature=0.3, max_tokens=1500)
            latency = time.time() - t0

            qa_items = extract_json(response["content"])
            print(f"   -> Section {s_idx + 1}/{len(selected_sections)}: Generated {len(qa_items)} Q&A pairs (Latency: {latency:.2f}s, Tokens: {response.get('total_tokens')})")

            for item in qa_items:
                if isinstance(item, dict) and "question" in item and "ground_truth" in item:
                    item_entry = {
                        "id": f"qa_{global_id:03d}",
                        "source_file": pdf_path.name,
                        "question": item["question"],
                        "ground_truth": item["ground_truth"],
                        "source_context_preview": section_text[:300] + "..."
                    }
                    all_generated_qa.append(item_entry)
                    global_id += 1

            time.sleep(2.0)

    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_generated_qa, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"✅ SUCCESS: Generated {len(all_generated_qa)} multi-hop Q&A pairs!")
    print(f"💾 Saved dataset to: {output_path.resolve()}")
    print("=" * 70)

    return all_generated_qa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic multi-hop Q&A pairs from PDFs.")
    parser.add_argument("--output", type=str, default="synthetic_golden_qa.json", help="Output JSON file path")
    parser.add_argument("--sections", type=int, default=2, help="Number of sections per PDF to process (default: 2)")
    args = parser.parse_args()

    generate_qa_from_pdfs(output_file=args.output, num_sections_per_pdf=args.sections)
