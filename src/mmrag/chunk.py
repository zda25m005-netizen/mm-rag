"""Chunking: split page texts into overlapping chunks.

Token counting uses a cheap whitespace approximation (~0.75 words/token is
close enough for chunk sizing; swap in tiktoken later if desired).
"""
from __future__ import annotations
import json, pathlib


def split_words(text: str) -> list[str]:
    return text.split()


def chunk_pages(processed_dir: str, chunk_tokens: int = 400,
                overlap_tokens: int = 80, min_chars: int = 200) -> int:
    proc = pathlib.Path(processed_dir)
    # tokens ~ words / 0.75  ->  words per chunk = chunk_tokens * 0.75
    words_per_chunk = int(chunk_tokens * 0.75)
    overlap_words = int(overlap_tokens * 0.75)
    step = max(words_per_chunk - overlap_words, 50)

    n = 0
    with open(proc / "pages.jsonl") as fin, open(proc / "chunks.jsonl", "w") as fout:
        for line in fin:
            rec = json.loads(line)
            words = split_words(rec["text"])
            if not words:
                continue
            for start in range(0, len(words), step):
                piece = " ".join(words[start:start + words_per_chunk])
                if len(piece) < min_chars and start > 0:
                    continue  # drop tiny tail fragments (keep short full pages)
                fout.write(json.dumps({
                    "chunk_id": f"{rec['doc_id']}_p{rec['page']}_c{start // step}",
                    "doc_id": rec["doc_id"],
                    "page": rec["page"],
                    "text": piece,
                }) + "\n")
                n += 1
                if start + words_per_chunk >= len(words):
                    break
    print(f"chunking done: {n} chunks -> chunks.jsonl")
    return n
