"""Figure index: Qdrant collection with named vectors (image + caption) and
hybrid search.

Each figure is stored once with two vectors:
    image   : CLIP embedding of the figure PNG   (from figure_vectors.npz)
    caption : CLIP text embedding of its caption (computed at build time)

Hybrid search embeds the query once, searches both vector spaces, and merges:
    score = alpha * sim_image + (1 - alpha) * sim_caption
alpha=1.0 -> image-only (Day 3 behavior); alpha=0.5 -> balanced hybrid.
"""
from __future__ import annotations
import json, pathlib
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class FigureIndex:
    def __init__(self, path: str = "data/qdrant_figures", collection: str = "mmrag_figures"):
        self.client = QdrantClient(path=path)
        self.collection = collection

    def build(self, processed_dir: str, clip_embedder, batch: int = 128) -> int:
        proc = pathlib.Path(processed_dir)
        meta = [json.loads(l) for l in open(proc / "figures.jsonl")]
        by_id = {m["figure_id"]: m for m in meta}

        data = np.load(proc / "figure_vectors.npz", allow_pickle=True)
        ids, img_vecs = [str(i) for i in data["ids"]], data["vectors"]
        dim = img_vecs.shape[1]

        # caption vectors via CLIP text encoder (same space as images)
        captions = [by_id[i]["caption"] if i in by_id else "" for i in ids]
        print(f"encoding {len(captions)} captions with CLIP text encoder...")
        cap_vecs = clip_embedder.encode_text(captions)

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={
                "image": VectorParams(size=dim, distance=Distance.COSINE),
                "caption": VectorParams(size=dim, distance=Distance.COSINE),
            },
        )
        points = []
        for j, fid in enumerate(ids):
            m = by_id.get(fid, {})
            points.append(PointStruct(
                id=j,
                vector={"image": img_vecs[j].tolist(), "caption": cap_vecs[j].tolist()},
                payload={"figure_id": fid, "doc_id": m.get("doc_id"),
                         "page": m.get("page"), "path": m.get("path"),
                         "caption": m.get("caption"),
                         "caption_source": m.get("caption_source")},
            ))
            if len(points) == batch:
                self.client.upsert(self.collection, points); points = []
        if points:
            self.client.upsert(self.collection, points)
        print(f"indexed {len(ids)} figures (image + caption vectors, dim={dim})")
        return len(ids)

    def search(self, query_vec: np.ndarray, top_k: int = 5, alpha: float = 0.5,
               pool: int = 30) -> list[dict]:
        """Hybrid search. alpha weights image similarity; (1-alpha) caption."""
        qv = query_vec.tolist()
        img_hits = self.client.query_points(self.collection, query=qv,
                                            using="image", limit=pool).points
        cap_hits = self.client.query_points(self.collection, query=qv,
                                            using="caption", limit=pool).points
        merged: dict[str, dict] = {}
        for h in img_hits:
            fid = h.payload["figure_id"]
            merged[fid] = {"img": h.score, "cap": 0.0, "payload": h.payload}
        for h in cap_hits:
            fid = h.payload["figure_id"]
            if fid in merged:
                merged[fid]["cap"] = h.score
            else:
                merged[fid] = {"img": 0.0, "cap": h.score, "payload": h.payload}
        scored = [
            {"score": round(alpha * v["img"] + (1 - alpha) * v["cap"], 4),
             "sim_image": round(v["img"], 4), "sim_caption": round(v["cap"], 4),
             **v["payload"]}
            for v in merged.values()
        ]
        scored.sort(key=lambda d: -d["score"])
        return scored[:top_k]
