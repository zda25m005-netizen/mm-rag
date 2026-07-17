"""CLIP embeddings for figures: images and text queries share one vector space.

Model: sentence-transformers/clip-ViT-B-32 (512-dim, CPU-friendly).
First run downloads the model (~600 MB) to ~/.cache — needs internet once.

Outputs data/processed/figure_vectors.npz:
    ids      : array of figure_id strings (aligned with vectors)
    vectors  : float32 [n_figures, 512], L2-normalized
"""
from __future__ import annotations
import json, pathlib
import numpy as np


class ClipEmbedder:
    def __init__(self, model_name: str = "clip-ViT-B-32", batch_size: int = 16):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.name = model_name

    def encode_images(self, paths: list[str]) -> np.ndarray:
        from PIL import Image
        vecs, batch, kept = [], [], []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                batch.append(img); kept.append(True)
            except Exception as e:
                print(f"  warn: cannot open {p}: {e}")
                kept.append(False)
                continue
            if len(batch) == self.batch_size:
                vecs.append(self.model.encode(batch, batch_size=self.batch_size,
                                              normalize_embeddings=True,
                                              show_progress_bar=False))
                for im in batch:
                    im.close()
                batch = []
        if batch:
            vecs.append(self.model.encode(batch, batch_size=self.batch_size,
                                          normalize_embeddings=True,
                                          show_progress_bar=False))
            for im in batch:
                im.close()
        out = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, 512), np.float32)
        return out, kept

    def encode_text(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False).astype(np.float32)


def embed_all_figures(processed_dir: str, out_name: str = "figure_vectors.npz") -> dict:
    proc = pathlib.Path(processed_dir)
    figures = [json.loads(l) for l in open(proc / "figures.jsonl")]
    paths = [f["path"] for f in figures]
    ids = [f["figure_id"] for f in figures]
    print(f"embedding {len(paths)} figures with CLIP (CPU — expect a few minutes)...")

    emb = ClipEmbedder()
    vectors, kept = emb.encode_images(paths)
    kept_ids = [i for i, k in zip(ids, kept) if k]
    assert len(kept_ids) == vectors.shape[0], "id/vector count mismatch"

    out = proc / out_name
    np.savez_compressed(out, ids=np.array(kept_ids), vectors=vectors)
    stats = {"figures_total": len(ids), "embedded": len(kept_ids),
             "failed": len(ids) - len(kept_ids), "dim": int(vectors.shape[1])}
    print(f"done: {stats} -> {out}")
    return stats
