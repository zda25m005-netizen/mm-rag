"""Day 5: RAGAS-style answer-quality eval over a QA set."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmrag.ragas_eval import evaluate_one

QUERIES = [
    "How does LoRA reduce the number of trainable parameters?",
    "What is the masked language model objective in BERT?",
    "What are the three steps of the RLHF pipeline?",
    "How does ColPali use late interaction over page images?",
    "What did they use to inject word-order information?",   # Day-2 hallucination case
    "How does the mixture-of-experts layer route tokens?",
]

if __name__ == "__main__":
    rows = []
    print(f"{'faith':>6} {'relev':>6} {'ctx_p':>6}  question")
    for q in QUERIES:
        r = evaluate_one(q)
        rows.append(r)
        print(f"{r['faithfulness']:>6} {r['answer_relevance']:>6} "
              f"{r['context_precision']:>6}  {q[:52]}")
    n = len(rows)
    print("-" * 74)
    for m in ("faithfulness", "answer_relevance", "context_precision"):
        print(f"avg {m:<18} {sum(r[m] for r in rows)/n:.3f}")
