"""Day 3: mine hard-negative training pairs for the reranker.

For each chunk: build a pseudo-query from its opening, retrieve top-K with bge,
positive = the source chunk, hard negatives = other high-ranked chunks.
Output data/processed/train_pairs.jsonl  rows: {query, passage, label}
"""
import json, pathlib, re, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_pipeline import make_pipeline, MODEL

PREFIX = "Represent this sentence for searching relevant passages: " if MODEL.startswith("BAAI/bge") else ""
CHUNKS = ROOT / "data/processed/chunks.jsonl"
OUT = ROOT / "data/processed/train_pairs.jsonl"

N_NEG = 3          # hard negatives per anchor
POOL = 12          # retrieve this many candidates
MIN_CHARS = 250


def pseudo_query(text: str) -> str:
    # first sentence, else first 25 words
    m = re.split(r"(?<=[.!?])\s+", text.strip())
    q = m[0] if m and len(m[0]) > 40 else " ".join(text.split()[:25])
    return q[:300]


def main():
    rag = make_pipeline()
    chunks = [json.loads(l) for l in open(CHUNKS)]
    chunks = [c for c in chunks if len(c["text"]) >= MIN_CHARS]
    n_pos = n_neg = 0
    with open(OUT, "w") as out:
        for c in chunks:
            q = pseudo_query(c["text"])
            qv = rag.embedder.encode([PREFIX + q])[0]
            hits = rag.index.search(qv, top_k=POOL)
            # positive
            out.write(json.dumps({"query": q, "passage": c["text"], "label": 1}) + "\n")
            n_pos += 1
            # hard negatives: high-ranked but not the anchor chunk
            negs = 0
            for h in hits:
                if h.get("chunk_id") == c["chunk_id"] or h["text"] == c["text"]:
                    continue
                out.write(json.dumps({"query": q, "passage": h["text"], "label": 0}) + "\n")
                n_neg += 1
                negs += 1
                if negs >= N_NEG:
                    break
    print(f"mined {n_pos} positives + {n_neg} hard negatives "
          f"({n_pos + n_neg} rows) -> {OUT}")
    # sample
    rows = [json.loads(l) for l in open(OUT)][:4]
    for r in rows:
        print(f"  [{r['label']}] q: {r['query'][:60]}")
        print(f"        p: {r['passage'][:60]}")


if __name__ == "__main__":
    main()
