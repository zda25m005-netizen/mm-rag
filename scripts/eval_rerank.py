"""Day 7: dense-only vs dense+reranker on the chunk-level gold set.

Three modes to isolate the value of YOUR fine-tuning:
  dense            - bge retrieval only
  dense+base       - bge -> off-the-shelf cross-encoder rerank
  dense+lora (v1)  - bge -> your LoRA-fine-tuned cross-encoder rerank
Correct = a top-k chunk is from the gold paper AND contains a gold phrase.
"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_pipeline import make_pipeline, MODEL
from mmrag.reranker import CrossEncoderReranker

GOLD = ROOT / "data/processed/gold_chunks.jsonl"
PREFIX = "Represent this sentence for searching relevant passages: " if MODEL.startswith("BAAI/bge") else ""
CANDIDATES = 50


def judge(hits, g, k):
    phrases = [p.lower() for p in g["accept_phrases"]]
    for h in hits[:k]:
        if h["doc_id"] == g["gold_doc"] and any(p in h["text"].lower() for p in phrases):
            return True
    return False


def rank_of(hits, g):
    phrases = [p.lower() for p in g["accept_phrases"]]
    for i, h in enumerate(hits):
        if h["doc_id"] == g["gold_doc"] and any(p in h["text"].lower() for p in phrases):
            return i + 1
    return None


def evaluate(rag, reranker, ks=(1, 3, 5)):
    gold = [json.loads(l) for l in open(GOLD)]
    hitk = {k: 0 for k in ks}
    mrr = 0.0
    for g in gold:
        qv = rag.embedder.encode([PREFIX + g["query"]])[0]
        hits = rag.index.search(qv, top_k=CANDIDATES if reranker else max(ks))
        if reranker:
            scores = reranker.score(g["query"], [h["text"] for h in hits])
            for h, s in zip(hits, scores):
                h["_s"] = s
            hits.sort(key=lambda h: -h["_s"])
        r = rank_of(hits, g)
        if r:
            mrr += 1.0 / r
        for k in ks:
            if judge(hits, g, k):
                hitk[k] += 1
    n = len(gold)
    return {**{f"R@{k}": round(hitk[k] / n, 3) for k in ks},
            "MRR": round(mrr / n, 3)}


if __name__ == "__main__":
    rag = make_pipeline()
    modes = [
        ("dense only", None),
        ("dense+base", CrossEncoderReranker(None)),
        ("dense+lora v1", CrossEncoderReranker(str(ROOT / "models/reranker_lora_v1"))),
    ]
    print("mode           R@1    R@3    R@5    MRR")
    for label, rr in modes:
        r = evaluate(rag, rr)
        print(f"{label:<14} {r['R@1']:<6} {r['R@3']:<6} {r['R@5']:<6} {r['MRR']}")
