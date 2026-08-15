import time

from verityrag.pipeline import VerityRAGPipeline
from verityrag.streaming import LiveDocumentStream

BASE_DOCS = [
    {"doc_id": "d1", "title": "Tesla",
     "text": "Nikola Tesla was a Serbian-American inventor and electrical engineer famous "
             "for his contributions to the design of the modern alternating current system."},
    {"doc_id": "d2", "title": "Warsaw",
     "text": "Warsaw is the capital and largest city of Poland, located on the Vistula river."},
]

LATE_DOC = {
    "doc_id": "d3", "title": "SteamEngine",
    "text": "The steam engine is an external combustion engine that converts heat into "
            "mechanical work, and it was central to the Industrial Revolution.",
}


def test_end_to_end_query_returns_grounded_answer():
    pipeline = VerityRAGPipeline()
    pipeline.ingest_documents(BASE_DOCS)
    result = pipeline.query("What is Nikola Tesla known for?")
    assert result.abstained is False
    assert result.answer
    assert result.grounding.overall_score >= 0.5


def test_pipeline_abstains_on_out_of_domain_query():
    pipeline = VerityRAGPipeline()
    pipeline.ingest_documents(BASE_DOCS)
    pipeline.agent.sufficiency_gate.threshold = 0.5  # demanding, since index is tiny
    result = pipeline.query("What was the primary cause of the 1973 oil crisis?")
    assert result.abstained is True


def test_document_not_searchable_before_streaming_in():
    pipeline = VerityRAGPipeline()
    pipeline.ingest_documents(BASE_DOCS)
    result_before = pipeline.query("How does a steam engine convert heat into work?")
    # SteamEngine doc not ingested yet -> should not be confidently grounded on it
    assert result_before.abstained is True or (
        result_before.evidence and result_before.evidence[0].doc_id != "d3"
    )


def test_new_document_becomes_searchable_after_streaming_ingestion():
    """The core 'real-time' property: a document that arrives via the live
    stream (not present at pipeline construction time) must become
    retrievable without any restart or manual reindex step."""
    pipeline = VerityRAGPipeline()
    pipeline.ingest_documents(BASE_DOCS)

    stream = LiveDocumentStream(pipeline, delay_seconds=0.01)
    stream.start()
    stream.enqueue(LATE_DOC)
    ingested_ok = stream.wait_until_ingested(1, timeout=5.0)
    stream.stop()

    assert ingested_ok is True
    result_after = pipeline.query("How does a steam engine convert heat into work?")
    assert result_after.abstained is False
    assert any(e.doc_id == "d3" for e in result_after.evidence)


def test_reranker_trains_and_plugs_into_pipeline():
    pipeline = VerityRAGPipeline()
    # train_reranker requires >=6 indexed chunks (see pipeline.py) -- a deliberate
    # floor so we don't bother training on a near-empty index -- so this test
    # needs a slightly bigger real corpus than the 2-doc BASE_DOCS to clear it.
    docs = BASE_DOCS + [LATE_DOC, {
        "doc_id": "d4", "title": "Huguenot",
        "text": "Huguenots were French Protestants who followed the Reformed tradition. "
                "Many emigrated from France to escape religious persecution in the "
                "sixteenth and seventeenth centuries.",
    }, {
        "doc_id": "d5", "title": "Martin Luther",
        "text": "Martin Luther was a German priest and theologian who began the "
                "Protestant Reformation in the sixteenth century. He challenged the "
                "authority of the Pope by teaching that salvation is received through faith.",
    }, {
        "doc_id": "d6", "title": "Southern California",
        "text": "Southern California is a geographic region of the U.S. state of "
                "California, home to Los Angeles and San Diego, known for its "
                "Mediterranean climate and dense population.",
    }]
    pipeline.ingest_documents(docs)
    trained = pipeline.train_reranker(n_queries=30)
    assert trained is True
    assert pipeline.agent.reranker_model is not None

    result = pipeline.query("What religion did Huguenots follow?")
    assert result.abstained is False
