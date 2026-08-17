"""
Tests SentenceTransformerEmbedder's WRAPPER LOGIC using a mocked
SentenceTransformer: dimension reporting (both old- and new-API method
names -- see embedding.py's hasattr fallback), empty-input handling, and
that normalize_embeddings=True is actually requested (cosine_sim_matrix
assumes pre-normalized input). None of this needs network access or a real
model download, so it runs the same in CI as anywhere else.

Gotcha worth flagging for future edits here: MagicMock() auto-creates ANY
attribute you touch, including ones you never configured -- so
hasattr(fake_model, "get_embedding_dimension") is True even on a mock meant
to represent an OLD sentence-transformers version that doesn't have that
method. Simulating "old API" requires `del fake_model.get_embedding_dimension`
to make hasattr correctly return False. This bit us for real: the first
version of these tests didn't do that, silently exercised the wrong code
path, and 2 of 3 tests failed once the fallback logic was added.

This does NOT verify the real model loads and produces sensible embeddings
end to end -- see test_sentence_embedder_live.py for that.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")


def _new_api_mock(dim: int) -> MagicMock:
    """Simulates a sentence-transformers version WITH get_embedding_dimension."""
    fake_model = MagicMock()
    fake_model.get_embedding_dimension.return_value = dim
    return fake_model


def _old_api_mock(dim: int) -> MagicMock:
    """Simulates a sentence-transformers version WITHOUT get_embedding_dimension
    (only the older get_sentence_embedding_dimension) -- deleting the
    attribute is required, not optional; see module docstring."""
    fake_model = MagicMock()
    del fake_model.get_embedding_dimension
    fake_model.get_sentence_embedding_dimension.return_value = dim
    return fake_model


def test_wrapper_uses_new_api_method_when_available():
    fake_model = _new_api_mock(384)
    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        from verityrag.embedding import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder("fake-model")
    assert emb.dim == 384
    fake_model.get_embedding_dimension.assert_called_once()


def test_wrapper_falls_back_to_old_api_method():
    fake_model = _old_api_mock(384)
    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        from verityrag.embedding import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder("fake-model")
    assert emb.dim == 384
    fake_model.get_sentence_embedding_dimension.assert_called_once()


def test_wrapper_requests_normalized_embeddings():
    fake_model = _new_api_mock(4)
    fake_model.encode.return_value = np.array([[0.5, 0.5, 0.5, 0.5]])
    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        from verityrag.embedding import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder("fake-model")
        result = emb.embed(["some text"])

    fake_model.encode.assert_called_once()
    _, kwargs = fake_model.encode.call_args
    # required: every downstream cosine_sim_matrix() call assumes this
    assert kwargs.get("normalize_embeddings") is True
    assert result.shape == (1, 4)


def test_wrapper_handles_empty_input_without_calling_model():
    fake_model = _new_api_mock(4)
    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        from verityrag.embedding import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder("fake-model")
        result = emb.embed([])

    assert result.shape == (0, 4)
    fake_model.encode.assert_not_called()
