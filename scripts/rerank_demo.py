"""Day 6: two-stage retrieval demo - dense-only vs dense+reranker."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_pipeline import make_pipeline, MODEL
from mmrag.reranker import CrossEncoderReranker, rerank_search

PREFIX = "Represent this sentence for searching relevant passages: " if MODEL.startswith("BAAI/bge") else ""
ADAPTER = str(ROOT / "models/reranker_lora_v1")

QUERIES = [
    "how does the model represent the order of tokens without recurrence",
    "keeping pretrained weights fixed while training small added parameters",
    "the reinforcement learning algorithm used to optimize against the reward",
]

if __name__ == "__main__":
    rag = make_pipeline()
    rr = CrossEncoderReranker(ADAPTER)
    print(f"reranker: {rr.name}\n")
    for q in QUERIES:
        print("=" * 74)
        print("Q:", q)
        qv = rag.embedder.encode([PREFIX + q])[0]
        dense = rag.index.search(qv, top_k=5)
        print("\n dense-only top-3:")
        for h in dense[:3]:
            print(f"   {h['doc_id']} p.{h['page']}  {h['text'][:65]}")
        reranked = rerank_search(rag, rr, q, candidates=50, top_k=5, prefix=PREFIX)
        print("\n reranked top-3:")
        for h in reranked[:3]:
            print(f"   {h['doc_id']} p.{h['page']}  rr={h['rerank_score']}  {h['text'][:55]}")
