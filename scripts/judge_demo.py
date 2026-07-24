"""Day 6: claim-level hallucination + citation-precision judge."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mmrag.judge import judge_answer

QUERIES = [
    "How does LoRA reduce the number of trainable parameters?",
    "What did they use to inject word-order information?",   # hallucination case
    "What are the three steps of the RLHF pipeline?",
    "How does the mixture-of-experts layer route tokens?",
]

if __name__ == "__main__":
    hrs, cps = [], []
    for q in QUERIES:
        r = judge_answer(q)
        print("=" * 74)
        print("Q:", q)
        for d in r["details"]:
            mark = "OK " if d["supported"] else "HALLUC"
            cite = {True: "cite:ok", False: "cite:WRONG", None: "cite:none"}[d["cited_ok"]]
            print(f"  [{mark:6}] [{cite:10}] {d['claim'][:60]}")
        print(f"  -> hallucination_rate={r['hallucination_rate']}  "
              f"citation_precision={r['citation_precision']}  ({r['n_claims']} claims)")
        hrs.append(r["hallucination_rate"])
        if r["citation_precision"] is not None:
            cps.append(r["citation_precision"])
    print("=" * 74)
    print(f"AVG hallucination_rate = {sum(hrs)/len(hrs):.3f}")
    print(f"AVG citation_precision = {sum(cps)/len(cps):.3f}" if cps else "no citations")
