"""
Hybrid retrieval: combines BM25 (lexical, catches exact terms/entities/numbers)
with dense cosine similarity (catches paraphrase/semantic matches), the same
pattern used by production hybrid search in Elasticsearch/OpenSearch.

Scores from the two systems live on different scales, so each is min-max
normalized over the candidate pool before combining -- a standard, defensible
approach (rather than adding raw BM25 and raw cosine directly, which are not
comparable).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embedding import Embedder, cosine_sim_matrix
from .indexing import LiveIndex


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    lexical_score: float       # raw BM25 score (unbounded, corpus-size dependent)
    dense_score: float         # raw cosine similarity in [0, 1] -- absolute, comparable
                                # across different queries/candidate pools. This is what
                                # the agent's sufficiency gate uses (see agent.py):
                                # unlike a pool-normalized score, "0.2" means the same
                                # thing regardless of how good/bad the rest of the pool is.
    hybrid_score: float        # blended RANKING score from pool-normalized components --
                                # only meaningful for ordering candidates within *this*
                                # retrieval call, not as an absolute confidence measure.


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def hybrid_retrieve(
    query: str,
    index: LiveIndex,
    embedder: Embedder,
    top_k: int = 8,
    lexical_weight: float = 0.4,
    dense_weight: float = 0.6,
) -> list[RetrievedChunk]:
    snapshot = index.snapshot()
    if not snapshot:
        return []

    lexical_scores = index.lexical_index.scores(query)
    query_vec = embedder.embed([query])
    doc_vecs = np.vstack([ic.vector for ic in snapshot])
    dense_scores = cosine_sim_matrix(query_vec, doc_vecs)[0]

    lexical_norm = _min_max_normalize(lexical_scores)
    dense_norm = _min_max_normalize(dense_scores)
    hybrid = lexical_weight * lexical_norm + dense_weight * dense_norm

    order = np.argsort(-hybrid)[:top_k]
    results = []
    for i in order:
        ic = snapshot[i]
        results.append(
            RetrievedChunk(
                chunk_id=ic.chunk.chunk_id,
                doc_id=ic.chunk.doc_id,
                title=ic.chunk.title,
                text=ic.chunk.text,
                lexical_score=float(lexical_scores[i]),
                dense_score=float(dense_scores[i]),
                hybrid_score=float(hybrid[i]),
            )
        )
    return results
