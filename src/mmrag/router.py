"""Query router: one ask() for both text RAG and figure search.

Routing (Day-5 scope): transparent keyword heuristic.
    figure-seeking -> FigureIndex hybrid search (returns figure paths)
    everything else -> text RAG pipeline (returns cited answer)
Upgrade path (later): small classifier or LLM-based routing; the interface
here stays the same, so the swap is one function.
"""
from __future__ import annotations
import re

FIGURE_PATTERNS = [
    r"\bfigure\b", r"\bfig\.?\b", r"\bdiagram\b", r"\bchart\b", r"\bplot\b",
    r"\bgraph\b", r"\bheatmap\b", r"\bvisuali[sz]ation\b", r"\bimage\b",
    r"\billustration\b", r"\bcurve[s]?\b", r"\bhistogram\b",
    r"\bshow\s+me\b", r"\bwhat\s+does\s+.*\s+look\s+like\b",
]
FIGURE_RE = re.compile("|".join(FIGURE_PATTERNS), re.IGNORECASE)


def classify_query(question: str) -> str:
    """Return 'figure' or 'text'."""
    return "figure" if FIGURE_RE.search(question) else "text"


class MultimodalRag:
    """Unified entry point. Lazy-loads each side on first use so text-only
    sessions never pay the CLIP model load."""

    def __init__(self, make_text_pipeline, make_figure_search):
        self._make_text = make_text_pipeline
        self._make_fig = make_figure_search
        self._text = None
        self._fig = None

    def ask(self, question: str, top_k: int = 5) -> dict:
        route = classify_query(question)
        if route == "figure":
            if self._fig is None:
                self._fig = self._make_fig()
            clip_emb, fig_index = self._fig
            qv = clip_emb.encode_text([question])[0]
            hits = fig_index.search(qv, top_k=top_k, alpha=0.5)
            return {
                "route": "figure",
                "question": question,
                "results": [
                    {"figure_id": h["figure_id"], "doc_id": h["doc_id"],
                     "page": h["page"], "path": h["path"],
                     "caption": h["caption"], "score": h["score"],
                     "sim_image": h["sim_image"], "sim_caption": h["sim_caption"]}
                    for h in hits
                ],
            }
        # text route
        if self._text is None:
            self._text = self._make_text()
        res = self._text.query(question)
        res["route"] = "text"
        return res
