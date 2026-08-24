import json
import time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from src.rag_engine import rag_engine
from src.llm import llm_client

EVAL_JUDGE_PROMPT = """You are an expert AI Evaluator assessing an Enterprise RAG system.
Given the User Question, Ground Truth Answer, Retrieved Contexts, and Generated Answer, compute the following metrics on a scale of 0.0 to 1.0:

1. **Faithfulness** (0.0 to 1.0): Are all factual claims in the generated answer completely supported by the retrieved context? (1.0 = zero hallucinations, 0.0 = completely fabricated).
2. **Context Precision** (0.0 to 1.0): Did the top retrieved contexts contain the precise facts needed to answer the question without irrelevant noise?
3. **Context Recall** (0.0 to 1.0): Did the retrieved contexts cover all the key facts present in the ground truth answer?
4. **Answer Relevancy** (0.0 to 1.0): Does the generated answer directly address the user question?

Respond ONLY in valid JSON format:
{
  "faithfulness": 1.0,
  "context_precision": 1.0,
  "context_recall": 1.0,
  "answer_relevancy": 1.0,
  "reasoning": "Brief explanation of the evaluation"
}
"""

class RAGEvaluator:
    """Automated Golden Dataset Evaluation Engine for CI/CD benchmarking."""

    def __init__(self, dataset_path: str = "golden_dataset.json"):
        self.dataset_path = Path(dataset_path)

    def load_dataset(self) -> List[Dict[str, Any]]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Golden dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        
        # 1. Run RAG Pipeline
        t0 = time.time()
        rag_output = rag_engine.query(question)
        eval_latency = time.time() - t0
        
        generated_answer = rag_output["answer"]
        contexts = [s["full_text"] for s in rag_output["sources"]]
        
        # 2. LLM Judge Evaluation
        eval_prompt = f"""User Question: {question}

Ground Truth Reference:
{ground_truth}

Retrieved Contexts:
{json.dumps(contexts, indent=2)}

Generated Answer:
{generated_answer}
"""
        messages = [
            {"role": "system", "content": EVAL_JUDGE_PROMPT},
            {"role": "user", "content": eval_prompt}
        ]
        
        judge_res = llm_client.generate(messages, temperature=0.0)
        content = judge_res["content"].strip()
        
        # Extract JSON from code fences if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        eval_status = "SUCCESS"
        try:
            # Handle potential JSON substrings within raw response
            if "{" in content and "}" in content:
                json_str = content[content.find("{"):content.rfind("}") + 1]
                scores = json.loads(json_str)
            else:
                scores = json.loads(content)
        except Exception as e:
            eval_status = "PARSE_ERROR"
            scores = {
                "faithfulness": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "answer_relevancy": 0.0,
                "reasoning": f"Judge parsing failed ({e}). Raw response: {content[:120]}..."
            }
            
        return {
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "faithfulness": float(scores.get("faithfulness", 0.0)),
            "context_precision": float(scores.get("context_precision", 0.0)),
            "context_recall": float(scores.get("context_recall", 0.0)),
            "answer_relevancy": float(scores.get("answer_relevancy", 0.0)),
            "reasoning": scores.get("reasoning", ""),
            "eval_status": eval_status,
            "latency_seconds": round(eval_latency, 3),
            "sources_count": len(rag_output["sources"])
        }

    def run_benchmark(self) -> Dict[str, Any]:
        items = self.load_dataset()
        results = []
        
        for item in items:
            res = self.evaluate_item(item)
            results.append(res)
            time.sleep(0.5)  # Rate pacing
            
        df = pd.DataFrame(results)
        valid_df = df[df["eval_status"] == "SUCCESS"]
        failed_count = int((df["eval_status"] != "SUCCESS").sum())
        
        summary = {
            "total_questions": len(results),
            "valid_evaluations": int(len(valid_df)),
            "failed_evaluations": failed_count,
            "avg_faithfulness": round(float(valid_df["faithfulness"].mean()), 3) if not valid_df.empty else 0.0,
            "avg_context_precision": round(float(valid_df["context_precision"].mean()), 3) if not valid_df.empty else 0.0,
            "avg_context_recall": round(float(valid_df["context_recall"].mean()), 3) if not valid_df.empty else 0.0,
            "avg_answer_relevancy": round(float(valid_df["answer_relevancy"].mean()), 3) if not valid_df.empty else 0.0,
            "avg_latency_seconds": round(float(df["latency_seconds"].mean()), 3),
            "detailed_results": results
        }
        return summary
