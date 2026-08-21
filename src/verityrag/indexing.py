"""
Incremental hybrid index (dense + lexical) designed for real-time updates.

Because HashingEmbedder is stateless, adding a document never requires
re-fitting anything -- we just embed the new chunk and append it. The BM25
lexical index is rebuilt on add (rank_bm25 has no incremental API), but this
is O(corpus tokens) and cheap enough at the "keep it live" scales this system
targets (thousands of chunks); a production deployment would swap in
Elasticsearch/OpenSearch for the lexical side, which *does* support true
incremental indexing -- the interface here is designed to make that swap a
drop-in change (see `LexicalIndex`).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .embedding import Embedder


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


@dataclass
class IndexedChunk:
    chunk: Chunk
    vector: np.ndarray
    indexed_at: float


class LexicalIndex:
    """BM25 lexical index. Swap-point for Elasticsearch/OpenSearch in
    production -- see elasticsearch_index.py's ElasticsearchLexicalIndex,
    which implements the same add_batch()/scores() interface but with
    TRUE incremental indexing, unlike this one (rank_bm25 has no
    incremental API, so add_batch() here still rebuilds internally on
    every call -- same cost as before, just renamed to match the shared
    interface both implementations now expose)."""

    def __init__(self):
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._bm25: BM25Okapi | None = None

    def add_batch(self, chunk_ids: list[str], texts: list[str]) -> None:
        self._chunk_ids.extend(chunk_ids)
        self._texts.extend(texts)
        tokenized = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def scores(self, query: str, chunk_ids: list[str]) -> np.ndarray:
        """`chunk_ids` is accepted for interface compatibility with
        ElasticsearchLexicalIndex (which genuinely needs it -- ES doesn't
        guarantee any particular result order beyond relevance ranking)
        but not used here: this class already tracks documents in the
        same append order as LiveIndex._chunks, since add_batch() is
        always called with new chunks in that same order, so its own
        internal ordering is already correct."""
        if self._bm25 is None:
            return np.zeros(0)
        return np.array(self._bm25.get_scores(_tokenize(query)))


class VectorIndex:
    """In-memory dense vector index. Swap-point for FAISS/pgvector at larger scale."""

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = np.zeros((0, dim))

    def add(self, vectors: np.ndarray):
        self._vectors = np.vstack([self._vectors, vectors]) if self._vectors.size else vectors

    def matrix(self) -> np.ndarray:
        return self._vectors


class LiveIndex:
    """
    Thread-safe, append-only hybrid index that supports adding documents
    while queries are being served concurrently -- the core requirement for
    "real-time": a document ingested now must be retrievable within the same
    process lifetime, with no restart / offline rebuild step.
    """

    def __init__(self, embedder: Embedder, lexical_index: LexicalIndex | None = None):
        self.embedder = embedder
        self.vector_index = VectorIndex(embedder.dim)
        self.lexical_index = lexical_index if lexical_index is not None else LexicalIndex()
        self._chunks: list[IndexedChunk] = []
        self._lock = threading.RLock()
        self.updates_count = 0

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        now = time.time()
        with self._lock:
            self.vector_index.add(vectors)
            for c, v in zip(chunks, vectors):
                self._chunks.append(IndexedChunk(chunk=c, vector=v, indexed_at=now))
            # Only the NEW chunks get passed to add_batch() -- true
            # incremental indexing when lexical_index is Elasticsearch-
            # backed. The rank_bm25 default still rebuilds internally on
            # every call regardless (see LexicalIndex.add_batch), same
            # cost as the old rebuild()-based design.
            self.lexical_index.add_batch([c.chunk_id for c in chunks], [c.text for c in chunks])
            self.updates_count += 1
        return len(chunks)

    def size(self) -> int:
        with self._lock:
            return len(self._chunks)

    def snapshot(self) -> list[IndexedChunk]:
        """Consistent read of all indexed chunks for a single query."""
        with self._lock:
            return list(self._chunks)

    def snapshot_with_matrix(self) -> tuple[list[IndexedChunk], np.ndarray]:
        """Like snapshot(), but also returns the corresponding dense vector
        matrix from the SAME read (same lock acquisition), so row i of the
        matrix is guaranteed to correspond to chunks[i] even if another
        thread adds documents concurrently between two separate calls.

        Exists because retrieval.py used to call snapshot() and then
        rebuild a full dense matrix from individual chunk.vector attributes
        via np.vstack -- reallocating and recopying the ENTIRE index's
        vectors into a brand new array on every single query. Found via a
        real MemoryError on a memory-constrained machine: 422MB reallocated
        per query against the full 845-chunk corpus, for data that
        vector_index already held in exactly this form. Use this instead."""
        with self._lock:
            return list(self._chunks), self.vector_index.matrix()