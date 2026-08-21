"""
Elasticsearch-backed lexical index -- the production swap-in for
LexicalIndex (indexing.py), which uses rank_bm25 and has no incremental
indexing API (forcing a full corpus rebuild on every single-document add,
~430ms/doc against a large index -- see README "Design decisions").
Elasticsearch supports true incremental indexing natively, which is the
actual point of this swap: add_batch() here only ever touches the NEW
documents passed to it, never rebuilds anything already indexed.

Optional dependency (`pip install elasticsearch`), same pattern as
sentence-transformers/anthropic/google-genai elsewhere in this project --
not imported at module load time, so nothing here adds a hard dependency
for users who don't need it.

Requires a running Elasticsearch instance -- not bundled, not started
automatically. Verified against the actual current elasticsearch-py
client API (9.5.0) by inspecting the installed package directly, not
assumed from training data, since client APIs do change between major
versions.
"""
from __future__ import annotations

import numpy as np

_MAPPING = {"properties": {"text": {"type": "text"}}}


class ElasticsearchLexicalIndex:
    """Drop-in replacement for LexicalIndex (indexing.py), backed by a
    real Elasticsearch instance instead of in-process rank_bm25.

    Usage:
        from verityrag.elasticsearch_index import ElasticsearchLexicalIndex
        from verityrag.indexing import LiveIndex
        live_index = LiveIndex(embedder=my_embedder, lexical_index=ElasticsearchLexicalIndex())

    Each instance targets one Elasticsearch index (default name
    "verityrag_chunks"). Calling this against an ES index that already
    has documents in it adds MORE documents alongside them, it does not
    start fresh -- call .reset() explicitly if you want an empty index
    (e.g. between separate test/demo runs against the same ES instance).
    """

    def __init__(self, es_url: str = "http://localhost:9200", index_name: str = "verityrag_chunks"):
        try:
            from elasticsearch import Elasticsearch
        except ImportError as e:
            raise ImportError(
                "ElasticsearchLexicalIndex requires the elasticsearch package, "
                "which isn't installed. Install it with: pip install elasticsearch"
            ) from e

        self.index_name = index_name
        self._client = Elasticsearch(es_url)
        if not self._client.indices.exists(index=self.index_name):
            self._client.indices.create(index=self.index_name, mappings=_MAPPING)

    def add_batch(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Indexes ONLY the new documents passed in -- true incremental
        indexing, the actual point of using Elasticsearch here. Never
        touches documents indexed by a previous call, unlike rank_bm25's
        add_batch() (indexing.py), which rebuilds everything every time."""
        if not chunk_ids:
            return
        for chunk_id, text in zip(chunk_ids, texts):
            self._client.index(index=self.index_name, id=chunk_id, document={"text": text})
        # Without an explicit refresh, a document indexed a moment ago
        # might not be visible to a search yet -- Elasticsearch's default
        # near-real-time refresh interval is ~1s. This project's whole
        # thesis is real-time ingestion: a newly-streamed document needs
        # to be searchable IMMEDIATELY, not after an arbitrary delay.
        self._client.indices.refresh(index=self.index_name)

    def scores(self, query: str, chunk_ids: list[str]) -> np.ndarray:
        """Returns one score per chunk_id, in the SAME order as the
        chunk_ids list passed in -- callers rely on this to align lexical
        scores positionally with other per-chunk data (e.g. dense
        vectors), and Elasticsearch itself gives no ordering guarantee
        beyond relevance ranking, so this has to be reconstructed
        explicitly rather than assumed. 0.0 for any chunk_id Elasticsearch
        doesn't return as a match at all (no term overlap) -- matching
        what rank_bm25 would also naturally produce for a non-match."""
        if not chunk_ids:
            return np.zeros(0)

        response = self._client.search(
            index=self.index_name,
            query={"match": {"text": query}},
            size=len(chunk_ids),
        )
        score_by_id = {hit["_id"]: hit["_score"] for hit in response["hits"]["hits"]}
        return np.array([score_by_id.get(cid, 0.0) for cid in chunk_ids])

    def reset(self) -> None:
        """Deletes and recreates the index -- explicit, not automatic on
        construction, since silently wiping existing data by default
        would be a surprising, destructive behavior for something that
        just looks like "connect to my index"."""
        if self._client.indices.exists(index=self.index_name):
            self._client.indices.delete(index=self.index_name)
        self._client.indices.create(index=self.index_name, mappings=_MAPPING)