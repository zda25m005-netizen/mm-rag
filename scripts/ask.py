"""Day 5: unified multimodal ask.

    python scripts/ask.py "How does LoRA reduce trainable parameters?"   # text
    python scripts/ask.py "show me the transformer architecture diagram" # figure
    python scripts/ask.py demo                                           # mixed set
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from mmrag.router import MultimodalRag, classify_query

ROOT = pathlib.Path(__file__).resolve().parents[1]


def make_text_pipeline():
    from run_pipeline import make_pipeline
    return make_pipeline()


def make_figure_search():
    from mmrag.clip_embed import ClipEmbedder
    from mmrag.figure_index import FigureIndex
    return ClipEmbedder(), FigureIndex(str(ROOT / "data/qdrant_figures"))


def show(res):
    print(f"\n[route: {res['route']}]  Q: {res['question']}")
    if res["route"] == "figure":
        for r, h in enumerate(res["results"][:3], 1):
            print(f"  {r}. {h['figure_id']}  score {h['score']} "
                  f"(img {h['sim_image']} / cap {h['sim_caption']})")
            print(f"     caption: {str(h['caption'])[:80]}")
            print(f"     file: {h['path']}")
    else:
        print(res["answer"][:600])
        for s in res["sources"][:3]:
            print(f"  source: {s['doc_id']} p.{s['page']} (score {s['score']})")


DEMO = [
    "show me the transformer architecture diagram",
    "What BLEU score did the Transformer achieve on English-to-German?",
    "attention heatmap figure",
    "How does LoRA reduce the number of trainable parameters?",
    "bar chart of benchmark scores across models",
    "What are the three steps of the RLHF pipeline?",
]

if __name__ == "__main__":
    rag = MultimodalRag(make_text_pipeline, make_figure_search)
    arg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "demo"
    if arg == "demo":
        for q in DEMO:
            print("=" * 74)
            print(f"router says: {classify_query(q)}")
            show(rag.ask(q))
    else:
        show(rag.ask(arg))
