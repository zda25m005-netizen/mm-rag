"""Week 1 driver: ingest -> chunk -> embed -> index -> query.

Usage:
    python scripts/run_pipeline.py build          # ingest + chunk + index
    python scripts/run_pipeline.py ask "..."      # one question
    python scripts/run_pipeline.py demo           # canned demo questions
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml
from mmrag.ingest import ingest_dir
from mmrag.chunk import chunk_pages
from mmrag.embed import TfidfEmbedder
from mmrag.index import ChunkIndex
from mmrag.rag import RagPipeline, get_generator

CFG = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1] / "configs/config.yaml"))
ROOT = pathlib.Path(__file__).resolve().parents[1]


def build():
    ingest_dir(ROOT / CFG["corpus"]["raw_dir"], ROOT / CFG["corpus"]["processed_dir"],
               ROOT / CFG["corpus"]["figures_dir"])
    chunk_pages(ROOT / CFG["corpus"]["processed_dir"],
                CFG["chunking"]["chunk_size_tokens"],
                CFG["chunking"]["overlap_tokens"],
                CFG["chunking"]["min_chunk_chars"])
    chunks_path = ROOT / CFG["corpus"]["processed_dir"] / "chunks.jsonl"
    texts = [json.loads(l)["text"] for l in open(chunks_path)]
    emb = TfidfEmbedder(str(ROOT / "data/processed/tfidf_state.pkl"))
    emb.fit(texts)
    idx = ChunkIndex(CFG["index"]["path"], CFG["index"]["collection"])
    idx.build(str(chunks_path), emb)


def make_pipeline():
    emb = TfidfEmbedder(str(ROOT / "data/processed/tfidf_state.pkl")).load()
    idx = ChunkIndex(CFG["index"]["path"], CFG["index"]["collection"])
    gen = get_generator(CFG["generator"]["provider"], CFG["generator"]["model"])
    return RagPipeline(emb, idx, gen, top_k=CFG["retrieval"]["top_k"])


def ask(q: str):
    rag = make_pipeline()
    res = rag.query(q)
    print(f"\nQ: {res['question']}\n\n{res['answer']}\n")
    print("sources:", *[f"  {s['doc_id']} p.{s['page']} (score {s['score']})"
                        for s in res["sources"]], sep="\n")


DEMO_QS = [
    "What BLEU score did the Transformer achieve on English-to-German translation?",
    "How does LoRA reduce the number of trainable parameters?",
    "What is the masked language model objective in BERT?",
    "How does ColPali use late interaction over page images?",
    "What does RLHF optimize with PPO?",
]

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "build":
        build()
    elif cmd == "ask":
        ask(" ".join(sys.argv[2:]))
    else:
        for q in DEMO_QS:
            ask(q)
            print("=" * 78)
