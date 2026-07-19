"""Pipeline driver: ingest -> chunk -> embed -> index -> query.

Embedder comes from configs/config.yaml (embedding.model):
    tfidf                  -> TfidfEmbedder (offline dev)
    BAAI/bge-small-en-v1.5 -> STEmbedder (neural, downloads once)

Usage:
    python scripts/run_pipeline.py build
    python scripts/run_pipeline.py ask "..."
    python scripts/run_pipeline.py demo
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml
from mmrag.ingest import ingest_dir
from mmrag.chunk import chunk_pages
from mmrag.embed import TfidfEmbedder, STEmbedder
from mmrag.index import ChunkIndex
from mmrag.rag import RagPipeline, get_generator

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/config.yaml"))
MODEL = CFG["embedding"]["model"]


def get_embedder(for_build: bool):
    if MODEL == "tfidf":
        emb = TfidfEmbedder(str(ROOT / "data/processed/tfidf_state.pkl"))
        if not for_build:
            emb.load()
        return emb
    return STEmbedder(MODEL, CFG["embedding"].get("batch_size", 32))


def collection_name():
    # separate collection per embedder so we can compare
    suffix = "tfidf" if MODEL == "tfidf" else MODEL.split("/")[-1].replace(".", "_")
    return f"{CFG['index']['collection']}_{suffix}"


def build():
    ingest_dir(ROOT / CFG["corpus"]["raw_dir"], ROOT / CFG["corpus"]["processed_dir"],
               ROOT / CFG["corpus"]["figures_dir"])
    chunk_pages(ROOT / CFG["corpus"]["processed_dir"],
                CFG["chunking"]["chunk_size_tokens"],
                CFG["chunking"]["overlap_tokens"],
                CFG["chunking"]["min_chunk_chars"])
    chunks_path = ROOT / CFG["corpus"]["processed_dir"] / "chunks.jsonl"
    emb = get_embedder(for_build=True)
    if MODEL == "tfidf":
        texts = [json.loads(l)["text"] for l in open(chunks_path)]
        emb.fit(texts)
    idx = ChunkIndex(CFG["index"]["path"], collection_name())
    idx.build(str(chunks_path), emb)
    print(f"built collection: {collection_name()} (model={MODEL})")


def make_pipeline():
    emb = get_embedder(for_build=False)
    idx = ChunkIndex(CFG["index"]["path"], collection_name())
    gen = get_generator(CFG["generator"]["provider"], CFG["generator"]["model"])
    return RagPipeline(emb, idx, gen, top_k=CFG["retrieval"]["top_k"])


def ask(q: str):
    res = make_pipeline().query(q)
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
            ask(q); print("=" * 78)
