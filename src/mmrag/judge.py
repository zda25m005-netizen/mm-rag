"""Claim-level LLM judge: hallucination rate + citation precision.

For each atomic claim in an answer:
  supported?  - is it backed by the retrieved context (any source)
  cited-ok?   - if it cites [doc_id p.N], does THAT source support it
"""
from __future__ import annotations
import re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mmrag.ragas_eval import answer_and_context, client

CITE_RE = re.compile(r"\[([0-9A-Za-z.]+)\s+p\.?\s*(\d+)\]")


def _chat(msg, max_tokens=200):
    r = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": msg}],
        max_tokens=max_tokens, temperature=0)
    return r.choices[0].message.content.strip()


def extract_claims(answer):
    out = _chat("Break the ANSWER into atomic factual claims, one per line. Keep any "
                "[doc_id p.N] citation attached to the claim it supports. No numbering, "
                "no extra text.\n\nANSWER:\n" + answer, max_tokens=300)
    return [l.strip("-*• ").strip() for l in out.splitlines() if l.strip()]


def _yes(msg):
    return _chat(msg + "\n\nAnswer only YES or NO.", max_tokens=3).upper().startswith("YES")


def supported(claim, ctx_str):
    return _yes(f"CONTEXT:\n{ctx_str}\n\nCLAIM: {claim}\n\nIs the CLAIM directly supported by the CONTEXT?")


def citation_ok(claim, ctx):
    cites = CITE_RE.findall(claim)
    if not cites:
        return None  # claim has no citation
    cited = [h for h in ctx if any(h["doc_id"] == d and str(h["page"]) == p for d, p in cites)]
    if not cited:
        return False  # cited a source that isn't even in the retrieved context
    ctext = "\n\n".join(h["text"][:800] for h in cited)
    return _yes(f"CITED SOURCE:\n{ctext}\n\nCLAIM: {claim}\n\nDoes the CITED SOURCE support the CLAIM?")


def judge_answer(question):
    answer, ctx, ctx_str = answer_and_context(question)
    claims = extract_claims(answer)
    details = []
    for c in claims:
        details.append({"claim": c, "supported": supported(c, ctx_str),
                        "cited_ok": citation_ok(c, ctx)})
    n = len(details) or 1
    unsupported = sum(1 for d in details if not d["supported"])
    cited = [d for d in details if d["cited_ok"] is not None]
    cite_correct = sum(1 for d in cited if d["cited_ok"])
    return {
        "question": question, "answer": answer, "details": details,
        "n_claims": len(details),
        "hallucination_rate": round(unsupported / n, 3),
        "citation_precision": round(cite_correct / len(cited), 3) if cited else None,
    }
