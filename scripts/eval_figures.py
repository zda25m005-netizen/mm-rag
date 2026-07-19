"""Day 7: figure-retrieval eval — image-only vs hybrid.

    python scripts/eval_figures.py
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mmrag.clip_embed import ClipEmbedder
from mmrag.figure_index import FigureIndex

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLD = ROOT / "data/processed/gold_figures.jsonl"


def evaluate(idx, emb, alpha, ks=(1, 3)):
    gold = [json.loads(l) for l in open(GOLD)]
    fig_hits = {k: 0 for k in ks}
    doc_hits = {k: 0 for k in ks}
    mrr = 0.0
    for g in gold:
        qv = emb.encode_text([g["query"]])[0]
        hits = idx.search(qv, top_k=max(ks), alpha=alpha)
        fids = [h["figure_id"] for h in hits]
        docs = [h["doc_id"] for h in hits]
        rank = next((i + 1 for i, f in enumerate(fids) if f in g["accept"]), None)
        if rank:
            mrr += 1.0 / rank
        for k in ks:
            if any(f in g["accept"] for f in fids[:k]):
                fig_hits[k] += 1
            if g["gold_doc"] in docs[:k]:
                doc_hits[k] += 1
    n = len(gold)
    return {
        "alpha": alpha,
        **{f"fig_recall@{k}": round(fig_hits[k] / n, 3) for k in ks},
        **{f"doc_recall@{k}": round(doc_hits[k] / n, 3) for k in ks},
        "fig_MRR": round(mrr / n, 3),
        "n": n,
    }


if __name__ == "__main__":
    emb = ClipEmbedder()
    idx = FigureIndex(str(ROOT / "data/qdrant_figures"))
    print("mode            fig_R@1  fig_R@3  doc_R@1  doc_R@3  fig_MRR")
    for label, a in (("image-only", 1.0), ("hybrid a=0.5", 0.5), ("caption-only", 0.0)):
        r = evaluate(idx, emb, a)
        print(f"{label:<15} {r['fig_recall@1']:<8} {r['fig_recall@3']:<8} "
              f"{r['doc_recall@1']:<8} {r['doc_recall@3']:<8} {r['fig_MRR']}")
