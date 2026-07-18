"""Day 4: build the figure index (image + caption vectors), compare
image-only vs hybrid search on test queries.

    python scripts/build_figure_index.py build
    python scripts/build_figure_index.py test
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mmrag.clip_embed import ClipEmbedder
from mmrag.figure_index import FigureIndex

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
IDX_PATH = str(ROOT / "data/qdrant_figures")

TEST_QUERIES = [
    "diagram of the transformer model architecture",
    "attention heatmap visualization",
    "bar chart comparing benchmark scores",
    "GPU memory usage comparison",
]

def build():
    emb = ClipEmbedder()
    idx = FigureIndex(IDX_PATH)
    idx.build(PROC, emb)

def test():
    emb = ClipEmbedder()
    idx = FigureIndex(IDX_PATH)
    for q in TEST_QUERIES:
        qv = emb.encode_text([q])[0]
        print("=" * 74)
        print(f"Q: {q}")
        for label, alpha in (("image-only (alpha=1.0)", 1.0), ("hybrid (alpha=0.5)", 0.5)):
            hits = idx.search(qv, top_k=3, alpha=alpha)
            print(f"\n  {label}:")
            for r, h in enumerate(hits, 1):
                print(f"    {r}. {h['figure_id']}  score {h['score']}"
                      f"  (img {h['sim_image']} / cap {h['sim_caption']})")
                print(f"       {str(h['caption'])[:80]}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    build() if cmd == "build" else test()
