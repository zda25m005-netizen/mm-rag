"""Day 3: embed all figures with CLIP, then run a sanity search.

    python scripts/embed_figures.py          # embed all figures (few minutes on CPU)
    python scripts/embed_figures.py test     # sanity: 3 text queries vs figure vectors
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from mmrag.clip_embed import ClipEmbedder, embed_all_figures

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"

TEST_QUERIES = [
    "diagram of the transformer model architecture",
    "attention heatmap visualization",
    "bar chart comparing benchmark scores",
]

def sanity_test():
    data = np.load(PROC / "figure_vectors.npz", allow_pickle=True)
    ids, vecs = data["ids"], data["vectors"]
    meta = {json.loads(l)["figure_id"]: json.loads(l) for l in open(PROC / "figures.jsonl")}
    emb = ClipEmbedder()
    qvecs = emb.encode_text(TEST_QUERIES)
    for q, qv in zip(TEST_QUERIES, qvecs):
        sims = vecs @ qv          # cosine (all normalized)
        top = np.argsort(-sims)[:3]
        print(f"\nQ: {q}")
        for rank, t in enumerate(top, 1):
            fid = str(ids[t]); m = meta.get(fid, {})
            print(f"  {rank}. {fid}  (sim {sims[t]:.3f})")
            print(f"     caption: {str(m.get('caption',''))[:90]}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        sanity_test()
    else:
        embed_all_figures(PROC)
