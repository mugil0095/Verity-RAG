from verityrag.chunking import chunk_document
from verityrag.embedding import HashingEmbedder
from verityrag.indexing import LiveIndex
from verityrag.retrieval import hybrid_retrieve

DOCS = [
    ("d1", "Tesla", "Nikola Tesla was a Serbian-American inventor famous for alternating current."),
    ("d2", "Warsaw", "Warsaw is the capital and largest city of Poland, on the Vistula river."),
    ("d3", "Steam engine", "The steam engine is an external combustion engine using steam as a working fluid."),
]


def _build_index():
    embedder = HashingEmbedder(n_features=4096)
    index = LiveIndex(embedder)
    for doc_id, title, text in DOCS:
        index.add_chunks(chunk_document(doc_id, title, text, max_tokens=60))
    return index, embedder


def test_retrieval_returns_at_most_k_results():
    index, embedder = _build_index()
    results = hybrid_retrieve("Who invented alternating current?", index, embedder, top_k=2)
    assert len(results) <= 2


def test_empty_index_returns_no_results():
    embedder = HashingEmbedder(n_features=1024)
    index = LiveIndex(embedder)
    results = hybrid_retrieve("anything", index, embedder, top_k=5)
    assert results == []


def test_relevant_document_ranks_first():
    index, embedder = _build_index()
    results = hybrid_retrieve("capital city of Poland", index, embedder, top_k=3)
    assert results[0].doc_id == "d2"


def test_scores_are_sorted_descending():
    index, embedder = _build_index()
    results = hybrid_retrieve("steam engine combustion", index, embedder, top_k=3)
    scores = [r.hybrid_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_lexical_keyword_match_surfaces_correct_doc():
    index, embedder = _build_index()
    results = hybrid_retrieve("Serbian American inventor", index, embedder, top_k=3)
    assert results[0].doc_id == "d1"
