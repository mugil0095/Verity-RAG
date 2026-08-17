"""
REAL end-to-end test for SentenceTransformerEmbedder -- actually downloads
and runs the model. Requires internet access and `pip install
sentence-transformers`.

NOT run as part of the main suite in the sandbox this project was built in
(no route to huggingface.co there -- see embedding.py). Run this locally
to confirm the swap-in genuinely works before relying on it:

    pytest tests/test_sentence_embedder_live.py -v -s
"""
import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

from src.verityrag.embedding import SentenceTransformerEmbedder, cosine_sim_matrix  # noqa: E402


@pytest.fixture(scope="module")
def embedder():
    try:
        return SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    except OSError as e:
        # Graceful skip instead of a hard error when the model can't be
        # downloaded (no internet, huggingface.co unreachable, etc.) -- an
        # environment issue, not a code failure, so it shouldn't look like
        # one in CI output.
        pytest.skip(f"Could not download the model (needs internet access): {e}")


def test_model_loads_and_reports_a_real_dimension(embedder):
    assert embedder.dim == 384  # all-MiniLM-L6-v2's known output size


def test_embeddings_are_l2_normalized(embedder):
    vecs = embedder.embed(["A sentence to check normalization."])
    norm = np.linalg.norm(vecs[0])
    assert abs(norm - 1.0) < 1e-4


def test_semantically_similar_sentences_score_higher_than_unrelated(embedder):
    """The actual point of this whole upgrade: real semantic understanding,
    not shared vocabulary. HashingEmbedder would likely score this LOW,
    since 'cat' and 'feline' share almost no literal characters -- a real
    encoder should recognize the paraphrase anyway."""
    a = "The cat sat on the mat."
    b = "A feline was resting on the rug."       # paraphrase, near-zero word overlap
    c = "Quarterly revenue exceeded forecasts."  # unrelated

    vecs = embedder.embed([a, b, c])
    sims = cosine_sim_matrix(vecs[0:1], vecs[1:3])[0]
    print(f"\nparaphrase similarity: {sims[0]:.3f} | unrelated similarity: {sims[1]:.3f}")
    assert sims[0] > sims[1]