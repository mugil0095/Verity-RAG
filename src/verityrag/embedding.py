"""
Embedding layer.

Design note (read this before assuming this is "just TF-IDF"):
This project intentionally does NOT download a pretrained neural embedding model
(e.g. sentence-transformers) because that requires reaching the HuggingFace Hub,
which is unavailable in this build environment. Rather than fake it, the embedding
layer uses scikit-learn's HashingVectorizer: a STATELESS bag-of-n-grams hashing
transform that needs no fitting on a corpus, no vocabulary, and no downloaded
weights. That statelessness is actually a good property for a *real-time* system:
new documents can be embedded and indexed the instant they arrive, with zero
retraining/refitting step -- something a fitted TF-IDF/SVD model can't do without
periodic re-fitting.

For production, swap `HashingEmbedder` for a real dense encoder (sentence-transformers,
OpenAI/Voyage/Anthropic embeddings API, etc.) behind the same `.embed()` interface --
the rest of the system (index, retrieval, agent, grounding) is agnostic to how the
vector was produced.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class Embedder:
    """Interface every embedder implementation follows."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """
    Streaming-safe embedder: bag-of-word/bigram hashing + L2 normalization,
    so cosine similarity == dot product.
    """

    def __init__(self, n_features: int = 2**16):
        self.dim = n_features
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            ngram_range=(1, 2),
            norm="l2",
            lowercase=True,
            stop_words="english",  # function words otherwise dominate short texts and
            # swamp the topical signal cosine similarity is supposed to capture --
            # see README "Design decisions" for the measurement that motivated this.
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim))
        mat = self._vectorizer.transform(texts)
        return mat.toarray()


def cosine_sim_matrix(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Both inputs assumed L2-normalized already (HashingEmbedder guarantees this),
    so cosine similarity is a plain dot product."""
    if query_vecs.size == 0 or doc_vecs.size == 0:
        return np.zeros((query_vecs.shape[0], doc_vecs.shape[0]))
    return query_vecs @ doc_vecs.T
