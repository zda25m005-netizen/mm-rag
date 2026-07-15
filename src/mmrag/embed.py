"""Pluggable embedders.

- TfidfEmbedder: offline dev embedder (no model download needed). Fits on the
  corpus, saves the vectorizer so queries embed consistently.
- STEmbedder: sentence-transformers (bge/e5) — use on Colab/GPU. Same interface.

Switch via config `embedding.model`: "tfidf" or an HF model name.
"""
from __future__ import annotations
import pathlib, pickle
import numpy as np


class TfidfEmbedder:
    name = "tfidf-svd-256"
    dim = 256

    def __init__(self, state_path: str = "data/processed/tfidf_state.pkl"):
        self.state_path = pathlib.Path(state_path)
        self.vec = None
        self.svd = None

    def fit(self, texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                                   sublinear_tf=True, stop_words="english")
        X = self.vec.fit_transform(texts)
        n_comp = min(self.dim, X.shape[1] - 1, len(texts) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=0)
        self.svd.fit(X)
        with open(self.state_path, "wb") as f:
            pickle.dump({"vec": self.vec, "svd": self.svd}, f)

    def load(self):
        with open(self.state_path, "rb") as f:
            s = pickle.load(f)
        self.vec, self.svd = s["vec"], s["svd"]
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        X = self.svd.transform(self.vec.transform(texts))
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, 1e-9)


class STEmbedder:
    """sentence-transformers embedder — for Colab/GPU runs."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def fit(self, texts):  # pretrained; nothing to fit
        pass

    def load(self):
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size,
                                 normalize_embeddings=True, show_progress_bar=False)


def get_embedder(model_name: str):
    if model_name == "tfidf":
        return TfidfEmbedder()
    return STEmbedder(model_name)
