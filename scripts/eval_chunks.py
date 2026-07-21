"""Chunk-level (passage) eval on paraphrased queries.

Correct = a top-k retrieved chunk is from the gold paper AND contains one of
the accept_phrases. Harder than doc-level -> has headroom for reranking.
Also compares bge with/without the recommended query prefix.
"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_pipeline import make_pipeline, MODEL

GOLD = ROOT / "data/processed/gold_chunks.jsonl"
PREFIX = "Represent this sentence for searching relevant passages: "


def evaluate(rag, use_prefix, ks=(1, 3, 5)):
    gold = [json.loads(l) for l in open(GOLD)]
    hits = {k: 0 for k in ks}
    mrr = 0.0
    for g in gold:
        q = (PREFIX + g["query"]) if use_prefix else g["query"]
        qv = rag.embedder.encode([q])[0]
        res = rag.index.search(qv, top_k=max(ks))
        phrases = [p.lower() for p in g["accept_phrases"]]

        def ok(h):
            return h["doc_id"] == g["gold_doc"] and any(p in h["text"].lower() for p in phrases)

        rank = next((i + 1 for i, h in enumerate(res) if ok(h)), None)
        if rank:
            mrr += 1.0 / rank
        for k in ks:
            if any(ok(h) for h in res[:k]):
                hits[k] += 1
    n = len(gold)
    return {**{f"chunk_R@{k}": round(hits[k] / n, 3) for k in ks},
            "MRR": round(mrr / n, 3), "n": n}


if __name__ == "__main__":
    rag = make_pipeline()
    print(f"model = {MODEL}")
    print("no prefix :", evaluate(rag, use_prefix=False))
    if MODEL.startswith("BAAI/bge"):
        print("bge prefix:", evaluate(rag, use_prefix=True))
