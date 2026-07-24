"""Day 7: validate the LLM judge against YOUR human labels.

For each claim: you see the context + claim, you label supported(1)/not(0).
The judge labels independently. We report agreement + Cohen's kappa.
Label based on the CONTEXT shown, not world knowledge.
"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mmrag.ragas_eval import answer_and_context  # for retrieval helpers
from mmrag.judge import supported, _chat  # noqa
from mmrag.reranker import CrossEncoderReranker
from run_pipeline import make_pipeline, MODEL

PREFIX = "Represent this sentence for searching relevant passages: " if MODEL.startswith("BAAI/bge") else ""

# (query used to fetch context, claim to judge). Mix of supported + fabricated.
ITEMS = [
    ("how does LoRA reduce trainable parameters", "LoRA freezes the pretrained weights and injects trainable low-rank matrices."),
    ("how does LoRA reduce trainable parameters", "LoRA was introduced by Google DeepMind in 2015."),
    ("masked language model objective in BERT", "BERT masks a subset of input tokens and predicts them."),
    ("masked language model objective in BERT", "BERT masks exactly 50 percent of the input tokens."),
    ("mixture of experts routes tokens to experts", "Each token is routed to a subset of the experts."),
    ("mixture of experts routes tokens to experts", "Every token is processed by all experts at once."),
    ("ColPali late interaction over page images", "ColPali produces multi-vector embeddings from page images."),
    ("ColPali late interaction over page images", "ColPali requires OCR to extract text before retrieval."),
    ("vision transformer splits image into patches", "ViT splits an image into fixed-size patches."),
    ("vision transformer splits image into patches", "ViT uses convolutional layers as its main building block."),
    ("how does the model encode word order", "Positional information is added to the token embeddings."),
    ("how does the model encode word order", "The model uses a recurrent network to encode order."),
]

_rag = _rr = None
def _ctx(query, k=3):
    global _rag, _rr
    if _rag is None:
        _rag, _rr = make_pipeline(), CrossEncoderReranker(None)
    qv = _rag.embedder.encode([PREFIX + query])[0]
    hits = _rag.index.search(qv, top_k=50)
    sc = _rr.score(query, [h["text"] for h in hits])
    for h, s in zip(hits, sc):
        h["_s"] = s
    hits.sort(key=lambda h: -h["_s"])
    ctx = hits[:k]
    return "\n\n".join(f"[{h['doc_id']} p.{h['page']}] {h['text'][:600]}" for h in ctx)


def kappa(h, j):
    n = len(h)
    po = sum(1 for a, b in zip(h, j) if a == b) / n
    ph1 = sum(h) / n; pj1 = sum(j) / n
    pe = ph1 * pj1 + (1 - ph1) * (1 - pj1)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0, po


if __name__ == "__main__":
    human, judge = [], []
    for i, (q, claim) in enumerate(ITEMS, 1):
        ctx = _ctx(q)
        print("\n" + "=" * 74)
        print(f"[{i}/{len(ITEMS)}] CONTEXT:\n{ctx[:700]}")
        print(f"\nCLAIM: {claim}")
        while True:
            ans = input("Supported by the context? (1=yes, 0=no): ").strip().lower()
            if ans in ("1", "y"):
                human.append(1); break
            if ans in ("0", "n"):
                human.append(0); break
        judge.append(1 if supported(claim, ctx) else 0)

    k, agree = kappa(human, judge)
    print("\n" + "=" * 74)
    print("idx  human  judge  match")
    for i, (h, j) in enumerate(zip(human, judge), 1):
        print(f"{i:>3}  {h:>5}  {j:>5}  {'OK' if h == j else 'X'}")
    print(f"\nagreement = {agree:.3f}   Cohen's kappa = {k:.3f}   (n={len(human)})")
    out = ROOT / "data/processed/judge_validation.json"
    out.write_text(json.dumps({"human": human, "judge": judge,
                               "agreement": agree, "kappa": k}, indent=2))
    print("saved", out)
