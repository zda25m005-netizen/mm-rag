"""Day 4: tool-calling demo - safe calculator verifies numeric claims."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmrag.agent import run_agent_tools, safe_calc

# unit check: the sandbox rejects non-arithmetic
try:
    safe_calc("__import__('os').system('echo hacked')")
    print("SECURITY FAIL")
except Exception:
    print("safe_calc correctly rejected non-arithmetic input\n")

QUERIES = [
    "LoRA can reduce trainable parameters by 10000x. If a model has 175 billion parameters, roughly how many are trainable with LoRA?",
    "What is the masked language model objective in BERT?",
]

if __name__ == "__main__":
    for q in QUERIES:
        print("=" * 74)
        print("Q:", q)
        out = run_agent_tools(q)
        print(" trace:")
        for t in out["trace"]:
            print("   -", t)
        print(" answer:", out["answer"][:400])
        print()
