"""Retrieval evaluation: recall@k and MRR on a labeled query set.

Gold file format (JSONL): {"question": ..., "gold_doc": "1706.03762"}
This grows into the Week-3/7 harness (nDCG, per-chunk labels, RAGAS).
"""
from __future__ import annotations
import json


def evaluate_retrieval(pipeline, gold_path: str, ks: tuple = (1, 3, 5)) -> dict:
    gold = [json.loads(l) for l in open(gold_path)]
    recalls = {k: 0 for k in ks}
    mrr_total = 0.0
    for g in gold:
        qvec = pipeline.embedder.encode([g["question"]])[0]
        hits = pipeline.index.search(qvec, top_k=max(ks))
        ranked_docs = [h["doc_id"] for h in hits]
        # first rank at which the gold doc appears (doc-level labels for Week 1)
        rank = next((i + 1 for i, d in enumerate(ranked_docs) if d == g["gold_doc"]), None)
        if rank:
            mrr_total += 1.0 / rank
            for k in ks:
                if rank <= k:
                    recalls[k] += 1
    n = len(gold)
    results = {f"recall@{k}": round(recalls[k] / n, 3) for k in ks}
    results["MRR"] = round(mrr_total / n, 3)
    results["n_queries"] = n
    return results
