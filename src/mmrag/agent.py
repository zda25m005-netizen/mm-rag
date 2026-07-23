"""LangGraph agent: Observe-Think-Act RAG loop.

retrieve -> grade (is context sufficient?) -> generate  OR  reformulate -> retrieve
Uses your dense+reranker retrieval and GPT-4o-mini for grading/generation.
"""
from __future__ import annotations
import pathlib, sys
from typing import TypedDict

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
        _rr = CrossEncoderReranker(None)   # production config: base cross-encoder
    return _rag, _rr


class State(TypedDict):
    question: str
    query: str
    context: list
    attempts: int
    sufficient: bool
    answer: str
    trace: list


def _chat(msg, max_tokens=200, temp=0.0):
    r = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": msg}],
        max_tokens=max_tokens, temperature=temp)
    return r.choices[0].message.content.strip()


def retrieve(state):
    rag, rr = _lazy()
    q = state["query"]
    qv = rag.embedder.encode([PREFIX + q])[0]
    hits = rag.index.search(qv, top_k=50)
    scores = rr.score(q, [h["text"] for h in hits])
    for h, s in zip(hits, scores):
        h["_s"] = s
    hits.sort(key=lambda h: -h["_s"])
    tr = state["trace"] + [f"retrieve(q='{q[:50]}') -> {hits[0]['doc_id']} p.{hits[0]['page']}"]
    return {"context": hits[:5], "attempts": state["attempts"] + 1, "trace": tr}


def grade(state):
    ctx = "\n\n".join(f"[{h['doc_id']} p.{h['page']}] {h['text'][:400]}" for h in state["context"])
    ans = _chat(f"Question: {state['question']}\n\nContext:\n{ctx}\n\n"
                "Can the question be fully answered from this context? Answer only YES or NO.",
                max_tokens=3)
    ok = ans.upper().startswith("YES")
    return {"sufficient": ok, "trace": state["trace"] + [f"grade -> {'sufficient' if ok else 'insufficient'}"]}


def reformulate(state):
    nq = _chat(f"The query '{state['query']}' did not retrieve enough to answer: "
               f"'{state['question']}'. Write one better, more specific search query. "
               "Output only the query.", max_tokens=40, temp=0.3)
    return {"query": nq, "trace": state["trace"] + [f"reformulate -> '{nq[:50]}'"]}


def generate(state):
    ctx = "\n\n".join(f"[{h['doc_id']} p.{h['page']}] {h['text'][:1200]}" for h in state["context"])
    ans = _chat(f"Answer using ONLY the context. Cite as [doc_id p.N].\n\nContext:\n{ctx}\n\n"
                f"Question: {state['question']}\n\nAnswer:", max_tokens=500, temp=0.1)
    return {"answer": ans, "trace": state["trace"] + ["generate"]}


def route(state):
    return "generate" if (state["sufficient"] or state["attempts"] >= 2) else "reformulate"


def build_agent():
    from langgraph.graph import StateGraph, END
    g = StateGraph(State)
    for name, fn in [("retrieve", retrieve), ("grade", grade),
                     ("reformulate", reformulate), ("generate", generate)]:
        g.add_node(name, fn)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route,
                            {"generate": "generate", "reformulate": "reformulate"})
    g.add_edge("reformulate", "retrieve")
    g.add_edge("generate", END)
    return g.compile()


def run_agent(question):
    app = build_agent()
    init = {"question": question, "query": question, "context": [],
            "attempts": 0, "sufficient": False, "answer": "", "trace": []}
    return app.invoke(init)
