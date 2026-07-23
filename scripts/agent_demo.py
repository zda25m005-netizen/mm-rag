"""Day 2: run the LangGraph agent and print its reasoning trace."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmrag.agent import run_agent

QUERIES = [
    "How does LoRA reduce the number of trainable parameters?",
    "What did they use to inject word-order information?",
]

if __name__ == "__main__":
    for q in QUERIES:
        print("=" * 74)
        print("Q:", q)
        out = run_agent(q)
        print("\n trace:")
        for step in out["trace"]:
            print("   -", step)
        print("\n answer:", out["answer"][:400])
        print()
