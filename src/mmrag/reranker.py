"""Cross-encoder reranker: base ms-marco-MiniLM + your LoRA adapter.

Scores (query, passage) pairs jointly - far more precise than bi-encoder
retrieval, so it re-ranks the top-N candidates from dense search.
"""
from __future__ import annotations
import torch

BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, adapter_path: str | None = None, max_length: int = 256):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(BASE)
        model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.eval()
        self.max_length = max_length
        self.name = f"reranker({'lora' if adapter_path else 'base'})"

    @torch.no_grad()
    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        enc = self.tok([query] * len(passages), passages, truncation=True,
                       max_length=self.max_length, padding=True, return_tensors="pt")
        logits = self.model(**enc).logits.squeeze(-1)
        return torch.sigmoid(logits).tolist()


def rerank_search(rag, reranker, query, candidates=50, top_k=5, prefix=""):
    """Two-stage: dense retrieve `candidates`, rerank, return top_k."""
    qv = rag.embedder.encode([prefix + query])[0]
    hits = rag.index.search(qv, top_k=candidates)
    scores = reranker.score(query, [h["text"] for h in hits])
    for h, s in zip(hits, scores):
        h["rerank_score"] = round(float(s), 4)
    hits.sort(key=lambda h: -h["rerank_score"])
    return hits[:top_k]
