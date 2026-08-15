from verityrag.chunking import chunk_document
from verityrag.embedding import HashingEmbedder
from verityrag.indexing import LiveIndex
from verityrag.retrieval import hybrid_retrieve, RetrievedChunk
from verityrag.reranker import generate_ict_training_data, train_reranker, rerank

TOPICS = [
    ("d1", "Tesla", "Nikola Tesla was a Serbian-American inventor and electrical engineer famous for his contributions to alternating current electricity."),
    ("d2", "Warsaw", "Warsaw is the capital and largest city of Poland, situated on the Vistula river in east-central Poland."),
    ("d3", "SteamEngine", "The steam engine is an external combustion engine that uses steam as its working fluid to perform mechanical work."),
    ("d4", "Teacher", "A teacher is a person who helps students acquire knowledge, competence, or virtue in a classroom setting."),
    ("d5", "SuperBowl", "Super Bowl 50 determined the champion of the National Football League for the 2015 season."),
    ("d6", "Huguenot", "Huguenots were French Protestants who followed the Reformed tradition of Protestantism."),
]


def _build_index():
    embedder = HashingEmbedder(n_features=4096)
    index = LiveIndex(embedder)
    for doc_id, title, text in TOPICS:
        index.add_chunks(chunk_document(doc_id, title, text, max_tokens=80))
    return index, embedder


def test_ict_training_data_generation_produces_examples():
    index, embedder = _build_index()
    snapshot = index.snapshot()
    examples = generate_ict_training_data(snapshot, embedder, n_queries=10, n_negatives=3)
    assert len(examples) > 0
    positives = [e for e in examples if e.label == 1]
    negatives = [e for e in examples if e.label == 0]
    assert len(positives) > 0 and len(negatives) > 0


def test_reranker_trains_without_error():
    index, embedder = _build_index()
    snapshot = index.snapshot()
    examples = generate_ict_training_data(snapshot, embedder, n_queries=20, n_negatives=3)
    model = train_reranker(examples)
    assert model is not None


def test_rerank_returns_same_candidate_set_reordered():
    index, embedder = _build_index()
    snapshot = index.snapshot()
    examples = generate_ict_training_data(snapshot, embedder, n_queries=20, n_negatives=3)
    model = train_reranker(examples)

    candidates = hybrid_retrieve("Serbian inventor alternating current", index, embedder, top_k=5)
    reranked = rerank("Serbian inventor alternating current", candidates, model)

    assert {c.chunk_id for c in reranked} == {c.chunk_id for c in candidates}
    assert len(reranked) == len(candidates)


def test_rerank_handles_empty_candidates():
    index, embedder = _build_index()
    snapshot = index.snapshot()
    examples = generate_ict_training_data(snapshot, embedder, n_queries=15, n_negatives=3)
    model = train_reranker(examples)
    assert rerank("irrelevant query", [], model) == []
