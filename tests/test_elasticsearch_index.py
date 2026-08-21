"""
Tests for elasticsearch_index.py. Mocked throughout -- never requires a
real, running Elasticsearch instance, same principle as
test_llm_providers.py mocking the Anthropic/Gemini SDKs instead of making
real API calls. This project's author doesn't have a live Elasticsearch
instance reachable from the environment these tests run in (it runs on a
different machine entirely), so a live test isn't practical here the way
test_sentence_embedder_live.py's HuggingFace download is.
"""
from unittest.mock import MagicMock, patch

import numpy as np

from verityrag.elasticsearch_index import ElasticsearchLexicalIndex


def _mock_client(index_exists=False):
    client = MagicMock()
    client.indices.exists.return_value = index_exists
    return client


def test_creates_index_on_construction_if_it_does_not_exist():
    client = _mock_client(index_exists=False)
    with patch("elasticsearch.Elasticsearch", return_value=client):
        ElasticsearchLexicalIndex(index_name="test_idx")

    client.indices.create.assert_called_once_with(
        index="test_idx",
        mappings={"properties": {"text": {"type": "text"}}},
    )


def test_does_not_recreate_index_if_it_already_exists():
    client = _mock_client(index_exists=True)
    with patch("elasticsearch.Elasticsearch", return_value=client):
        ElasticsearchLexicalIndex(index_name="test_idx")

    client.indices.create.assert_not_called()


def test_add_batch_indexes_only_the_new_documents_passed_in():
    """The actual point of this class: add_batch() must never touch
    anything beyond what's passed to THIS call -- true incremental
    indexing, unlike LexicalIndex's rebuild-everything-every-time
    default. Confirmed by checking exactly what got indexed, not just
    that indexing happened."""
    client = _mock_client(index_exists=True)
    with patch("elasticsearch.Elasticsearch", return_value=client):
        idx = ElasticsearchLexicalIndex(index_name="test_idx")
        idx.add_batch(["c1", "c2"], ["first chunk text", "second chunk text"])

    assert client.index.call_count == 2
    client.index.assert_any_call(index="test_idx", id="c1", document={"text": "first chunk text"})
    client.index.assert_any_call(index="test_idx", id="c2", document={"text": "second chunk text"})


def test_add_batch_refreshes_so_new_documents_are_immediately_searchable():
    """Elasticsearch's default ~1s refresh interval would mean a
    just-streamed document isn't searchable right away -- this project's
    whole thesis is real-time ingestion, so this must be explicit, not
    left to chance."""
    client = _mock_client(index_exists=True)
    with patch("elasticsearch.Elasticsearch", return_value=client):
        idx = ElasticsearchLexicalIndex(index_name="test_idx")
        idx.add_batch(["c1"], ["some text"])

    client.indices.refresh.assert_called_once_with(index="test_idx")


def test_add_batch_with_empty_list_does_nothing():
    client = _mock_client(index_exists=True)
    with patch("elasticsearch.Elasticsearch", return_value=client):
        idx = ElasticsearchLexicalIndex(index_name="test_idx")
        idx.add_batch([], [])

    client.index.assert_not_called()
    client.indices.refresh.assert_not_called()


def test_scores_returns_values_in_the_same_order_as_chunk_ids_passed_in():
    """The actual point of this method: Elasticsearch gives no ordering
    guarantee beyond relevance ranking, so the returned array must be
    explicitly reconstructed to match chunk_ids' order, not just contain
    the right VALUES in whatever order ES happened to return them."""
    client = _mock_client(index_exists=True)
    client.search.return_value = {
        "hits": {"hits": [
            {"_id": "c3", "_score": 5.0},
            {"_id": "c1", "_score": 2.0},
            # c2 deliberately absent -- no term overlap, should default to 0.0
        ]}
    }
    with patch("elasticsearch.Elasticsearch", return_value=client):
        idx = ElasticsearchLexicalIndex(index_name="test_idx")
        scores = idx.scores("some query", ["c1", "c2", "c3"])

    assert isinstance(scores, np.ndarray)
    assert list(scores) == [2.0, 0.0, 5.0]  # matches chunk_ids order: c1, c2, c3


def test_scores_calls_search_with_correct_query_and_size():
    client = _mock_client(index_exists=True)
    client.search.return_value = {"hits": {"hits": []}}
    with patch("elasticsearch.Elasticsearch", return_value=client):
        idx = ElasticsearchLexicalIndex(index_name="test_idx")
        idx.scores("Nikola Tesla", ["c1", "c2", "c3", "c4"])

    client.search.assert_called_once_with(
        index="test_idx",
        query={"match": {"text": "Nikola Tesla"}},
        size=4,  # matches len(chunk_ids), not some fixed/default value
    )


def test_scores_with_empty_chunk_ids_returns_empty_array_without_querying():
    client = _mock_client(index_exists=True)
    with patch("elasticsearch.Elasticsearch", return_value=client):
        idx = ElasticsearchLexicalIndex(index_name="test_idx")
        scores = idx.scores("anything", [])

    assert scores.size == 0
    client.search.assert_not_called()


def test_reset_deletes_and_recreates_the_index():
    client = _mock_client(index_exists=True)
    with patch("elasticsearch.Elasticsearch", return_value=client):
        idx = ElasticsearchLexicalIndex(index_name="test_idx")
        client.indices.create.reset_mock()  # clear the constructor's own create call
        idx.reset()

    client.indices.delete.assert_called_once_with(index="test_idx")
    client.indices.create.assert_called_once_with(
        index="test_idx",
        mappings={"properties": {"text": {"type": "text"}}},
    )


def test_raises_clear_error_when_elasticsearch_package_not_installed():
    """Simulates the package genuinely missing via sys.modules, not a
    reload dance -- ElasticsearchLexicalIndex does the `import
    elasticsearch` lazily inside __init__ (not at module level), so
    patching sys.modules around just the constructor call is sufficient
    and more direct than reloading the whole module."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "elasticsearch":
            raise ImportError("No module named 'elasticsearch'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        import pytest
        with pytest.raises(ImportError, match="pip install elasticsearch"):
            ElasticsearchLexicalIndex()


def test_integrates_with_live_index_as_a_pluggable_lexical_backend():
    """The actual integration point: LiveIndex (indexing.py) accepting
    this as its lexical_index, exactly as a real pipeline would use it,
    with a real HashingEmbedder alongside it (only the lexical side is
    mocked -- the dense/embedding side is completely real here)."""
    from verityrag.chunking import chunk_document
    from verityrag.embedding import HashingEmbedder
    from verityrag.indexing import LiveIndex

    client = _mock_client(index_exists=True)
    client.search.return_value = {"hits": {"hits": [{"_id": "c1", "_score": 3.0}]}}

    with patch("elasticsearch.Elasticsearch", return_value=client):
        es_lexical = ElasticsearchLexicalIndex(index_name="test_idx")
        live_index = LiveIndex(embedder=HashingEmbedder(n_features=2048), lexical_index=es_lexical)

        chunks = chunk_document("d1", "Test Doc", "Some real text about Nikola Tesla.", max_tokens=50)
        live_index.add_chunks(chunks)

        assert client.index.call_count == len(chunks)  # each chunk actually got indexed via ES

        chunk_ids = [ic.chunk.chunk_id for ic in live_index.snapshot()]
        scores = live_index.lexical_index.scores("Tesla", chunk_ids)
        assert scores.size == len(chunk_ids)


def test_integrates_with_verityrag_pipeline_end_to_end():
    """One level up from the LiveIndex test above -- confirms
    VerityRAGPipeline's lexical_index= constructor param actually reaches
    all the way through to the index it builds, which is the level a real
    user would actually interact with."""
    from verityrag.pipeline import VerityRAGPipeline

    client = _mock_client(index_exists=True)
    client.search.return_value = {"hits": {"hits": []}}

    with patch("elasticsearch.Elasticsearch", return_value=client):
        es_lexical = ElasticsearchLexicalIndex(index_name="test_idx")
        pipeline = VerityRAGPipeline(lexical_index=es_lexical)

        assert pipeline.index.lexical_index is es_lexical

        pipeline.ingest_document("d1", "Test Doc", "Some real text about Nikola Tesla.")
        assert client.index.called  # ingestion actually reached the ES-backed index