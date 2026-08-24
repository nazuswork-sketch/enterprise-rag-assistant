import sys
import io
import json
import argparse
from pathlib import Path

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'E:/project1')

from src.evaluation import RAGEvaluator

def main():
    parser = argparse.ArgumentParser(description="Run Enterprise RAG CI/CD Evaluation Benchmark.")
    parser.add_argument("--dataset", type=str, default="synthetic_golden_qa.json", help="Path to golden dataset JSON")
    parser.add_argument("--output", type=str, default="eval_report.json", help="Path to output report JSON")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        dataset_path = Path("golden_dataset.json")
        
    print("=" * 70)
    print(">> ENTERPRISE RAG CI/CD EVALUATION BENCHMARK")
    print(f">> Evaluating Dataset: {dataset_path.name}")
    print("=" * 70)
    
    evaluator = RAGEvaluator(str(dataset_path))
    results = evaluator.run_benchmark()
    
    print("\n--- BENCHMARK RESULTS SUMMARY ---")
    print(f"Total Evaluated Questions: {results['total_questions']}")
    print(f"Average Faithfulness (Hallucination Guard): {results['avg_faithfulness']:.2f} / 1.00")
    print(f"Average Context Precision:                 {results['avg_context_precision']:.2f} / 1.00")
    print(f"Average Context Recall:                    {results['avg_context_recall']:.2f} / 1.00")
    print(f"Average Answer Relevancy:                  {results['avg_answer_relevancy']:.2f} / 1.00")
    print(f"Average Pipeline Latency:                  {results['avg_latency_seconds']:.2f} seconds")
    print("-" * 70)
    
    min_faithfulness = 0.80
    min_recall = 0.75
    
    passed = (results['avg_faithfulness'] >= min_faithfulness) and (results['avg_context_recall'] >= min_recall)
    
    if passed:
        print("[PASSED] CI/CD BENCHMARK: All accuracy and faithfulness gates met.")
    else:
        print("[WARNING] CI/CD BENCHMARK: Scores below required enterprise threshold.")
        
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed report exported to '{args.output}'")

if __name__ == '__main__':
    main()
