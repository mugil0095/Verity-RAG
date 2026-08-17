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
vector was produced. SentenceTransformerEmbedder below is that swap-in.
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


class SentenceTransformerEmbedder(Embedder):
    """
    Real dense neural embeddings via sentence-transformers -- the swap-in
    this module's docstring above points to. Requires `pip install
    sentence-transformers` and, once, normal internet access: the model
    downloads from the HuggingFace Hub on first use, then runs fully
    offline/locally after that (no per-call network access, no API key).

    BUILD-ENVIRONMENT NOTE: the sandbox this project was developed in has no
    route to huggingface.co (see the module docstring), so this class's
    *logic* is covered by mocked unit tests (tests/test_sentence_embedder.py)
    but the real download-and-encode path has NOT been run end to end in
    that sandbox -- only tested for real on a machine with normal internet
    access. Run tests/test_sentence_embedder_live.py locally to confirm.

    `normalize_embeddings=True` matters beyond just being tidy: every
    consumer of embeddings in this codebase (retrieval.py, grounding.py,
    sufficiency.py) goes through cosine_sim_matrix() above, which assumes
    pre-normalized input and just takes a dot product -- skip normalization
    here and every downstream similarity score would be silently wrong.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim))
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )