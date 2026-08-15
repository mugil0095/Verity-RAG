from verityrag.chunking import chunk_document
from verityrag.embedding import HashingEmbedder
from verityrag.indexing import LiveIndex


def _sample_chunks(doc_id="d1", title="Sample"):
    text = ("The Eiffel Tower is a wrought-iron lattice tower in Paris. "
            "It was completed in 1889 and named after engineer Gustave Eiffel. "
            "It is one of the most visited monuments in the world.")
    return chunk_document(doc_id, title, text, max_tokens=25)


def test_incremental_add_increases_index_size():
    index = LiveIndex(HashingEmbedder(n_features=2048))
    assert index.size() == 0
    added = index.add_chunks(_sample_chunks())
    assert added > 0
    assert index.size() == added


def test_multiple_adds_accumulate():
    index = LiveIndex(HashingEmbedder(n_features=2048))
    index.add_chunks(_sample_chunks("d1"))
    size_after_first = index.size()
    index.add_chunks(_sample_chunks("d2"))
    assert index.size() > size_after_first


def test_lexical_index_finds_keyword_match():
    # BM25's IDF term can legitimately go negative on a tiny corpus where a
    # query term appears in most/all chunks -- that's correct BM25 behavior,
    # not a bug -- so this checks RELATIVE ranking, not an absolute score floor.
    index = LiveIndex(HashingEmbedder(n_features=2048))
    index.add_chunks(_sample_chunks())  # Eiffel Tower passage
    index.add_chunks(chunk_document("d2", "Unrelated", "Photosynthesis converts sunlight into chemical energy in plants."))
    scores = index.lexical_index.scores("Gustave Eiffel engineer tower")
    assert scores.size == index.size()
    eiffel_chunk_positions = [i for i, ic in enumerate(index.snapshot()) if ic.chunk.doc_id == "d1"]
    other_positions = [i for i, ic in enumerate(index.snapshot()) if ic.chunk.doc_id == "d2"]
    assert max(scores[i] for i in eiffel_chunk_positions) > max(scores[i] for i in other_positions)


def test_empty_index_lexical_scores_empty():
    index = LiveIndex(HashingEmbedder(n_features=2048))
    scores = index.lexical_index.scores("anything")
    assert scores.size == 0


def test_snapshot_reflects_current_state():
    index = LiveIndex(HashingEmbedder(n_features=2048))
    index.add_chunks(_sample_chunks())
    snap1 = index.snapshot()
    index.add_chunks(_sample_chunks("d2"))
    snap2 = index.snapshot()
    assert len(snap2) > len(snap1)
