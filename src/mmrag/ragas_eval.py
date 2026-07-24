"""RAGAS-style answer-quality metrics via LLM judge (GPT-4o-mini).

faithfulness      - fraction of answer claims supported by context (low = hallucination)
answer_relevance  - does the answer address the question
context_precision - fraction of retrieved passages that are relevant
"""
from __future__ import annotations
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_pipeline import make_pipeline, MODEL
from mmrag.reranker import CrossEncoderReranker
from openai import OpenAI

PREFIX = "Represent this sentence for searching relevant passages: " if MODEL.startswith("BAAI/bge") else ""
client = OpenAI()
_rag = _rr = None


def _lazy():
    global _rag, _rr
    if _rag is None:
        _rag = make_pipeline()
        _rr = CrossEncoderReranker(None)
    return _rag, _rr


def _score(prompt) -> float:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt + "\n\nOutput ONLY a number between 0.0 and 1.0."}],
        max_tokens=6, temperature=0)
    m = re.search(r"[01](?:\.\d+)?", r.choices[0].message.content)
    return float(m.group()) if m else 0.0


def answer_and_context(question, k=5):
    rag, rr = _lazy()
    qv = rag.embedder.encode([PREFIX + question])[0]
    hits = rag.index.search(qv, top_k=50)
    scores = rr.score(question, [h["text"] for h in hits])
    for h, s in zip(hits, scores):
        h["_s"] = s
    hits.sort(key=lambda h: -h["_s"])
    ctx = hits[:k]
    ctx_str = "\n\n".join(f"[{h['doc_id']} p.{h['page']}] {h['text'][:1000]}" for h in ctx)
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content":
            f"Answer using ONLY the context. Cite [doc_id p.N]. If context is "
            f"insufficient, say so.\n\nContext:\n{ctx_str}\n\nQuestion: {question}\n\nAnswer:"}],
        max_tokens=400, temperature=0.1)
    return r.choices[0].message.content, ctx, ctx_str


def evaluate_one(question):
    answer, ctx, ctx_str = answer_and_context(question)
    faith = _score(f"CONTEXT:\n{ctx_str}\n\nANSWER:\n{answer}\n\n"
                   "What fraction of the factual claims in the ANSWER are directly "
                   "supported by the CONTEXT?")
    rel = _score(f"QUESTION: {question}\n\nANSWER:\n{answer}\n\n"
                 "How well does the ANSWER address the QUESTION?")
    prec = _score(f"QUESTION: {question}\n\nRETRIEVED PASSAGES:\n{ctx_str}\n\n"
                  "What fraction of the passages are relevant to answering the question?")
    return {"question": question, "answer": answer,
            "faithfulness": faith, "answer_relevance": rel, "context_precision": prec}
