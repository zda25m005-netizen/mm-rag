"""Day 3: query decomposition demo."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmrag.agent import run_agent_decompose

QUERIES = [
    "Compare how LoRA and prompt tuning each reduce the number of trainable parameters.",
    "What is the masked language model objective in BERT?",
]

if __name__ == "__main__":
    for q in QUERIES:
        print("=" * 74)
        print("Q:", q)
        out = run_agent_decompose(q)
        print("\n sub-questions:")
        for s in out["subquestions"]:
            print("   -", s)
        print("\n trace:")
        for t in out["trace"]:
            print("   -", t)
        print("\n papers used:", sorted({h["doc_id"] for h in out["context"]}))
        print("\n answer:", out["answer"][:500])
        print()
