"""Qdrant index (local file mode — no server needed; same API as the real thing)."""
from __future__ import annotations
import json, pathlib
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class ChunkIndex:
    def __init__(self, path: str = "data/qdrant", collection: str = "mmrag_chunks"):
        self.client = QdrantClient(path=path)
        self.collection = collection

    def build(self, chunks_path: str, embedder, batch: int = 256):
        chunks = [json.loads(l) for l in open(chunks_path)]
        texts = [c["text"] for c in chunks]
        dim = embedder.encode(["probe"]).shape[1]
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        for i in range(0, len(chunks), batch):
            vecs = embedder.encode(texts[i:i + batch])
            points = [
                PointStruct(id=i + j, vector=vecs[j].tolist(), payload=chunks[i + j])
                for j in range(len(vecs))
            ]
            self.client.upsert(self.collection, points)
        print(f"indexed {len(chunks)} chunks into '{self.collection}' (dim={dim})")
        return len(chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        hits = self.client.query_points(
            self.collection, query=query_vec.tolist(), limit=top_k
        ).points
        return [
            {"score": round(h.score, 4), **h.payload} for h in hits
        ]
