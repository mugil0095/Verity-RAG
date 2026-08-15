import numpy as np

from verityrag.embedding import HashingEmbedder, cosine_sim_matrix


def test_embedding_dimension_consistent():
    emb = HashingEmbedder(n_features=1024)
    vecs = emb.embed(["Nikola Tesla invented the AC motor.", "Warsaw is the capital of Poland."])
    assert vecs.shape == (2, 1024)


def test_embedding_empty_input():
    emb = HashingEmbedder(n_features=1024)
    vecs = emb.embed([])
    assert vecs.shape == (0, 1024)


def test_similar_texts_more_similar_than_dissimilar():
    emb = HashingEmbedder(n_features=4096)
    a = "Nikola Tesla was a physicist and electrical engineer."
    b = "Tesla, an electrical engineer, worked as a physicist."
    c = "Warsaw hosts an annual international marathon in autumn."

    vecs = emb.embed([a, b, c])
    sims = cosine_sim_matrix(vecs[0:1], vecs[1:3])[0]
    sim_ab, sim_ac = sims[0], sims[1]
    assert sim_ab > sim_ac


def test_embedding_is_deterministic():
    emb = HashingEmbedder(n_features=2048)
    v1 = emb.embed(["Repeated input text for determinism check."])
    v2 = emb.embed(["Repeated input text for determinism check."])
    assert np.allclose(v1, v2)


def test_vectors_are_unit_normalized():
    emb = HashingEmbedder(n_features=2048)
    vecs = emb.embed(["Some non-trivial sentence with several distinct words."])
    norm = np.linalg.norm(vecs[0])
    assert abs(norm - 1.0) < 1e-6
