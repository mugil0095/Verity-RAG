"""
Learning-to-rank reranker (LightGBM LGBMRanker) over the hybrid-retrieval
candidate set.

Training data: production rerankers are usually trained on real click/relevance
judgments, which this project doesn't have. Instead we bootstrap with a classic
weak-supervision technique -- the Inverse Cloze Task (ICT): sample a sentence
out of a chunk to use as a synthetic "query," treat its source chunk as the
positive, and treat other random chunks as negatives. This is a real, published
self-supervision method (used e.g. in the ORQA / REALM retrieval literature),
not a placeholder -- and it means the reranker can be (re)trained on whatever
corpus is currently indexed, including one that grew via real-time ingestion.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from .chunking import split_sentences
from .embedding import Embedder, cosine_sim_matrix
from .indexing import IndexedChunk
from .retrieval import RetrievedChunk


FEATURE_NAMES = [
    "hybrid_score",
    "lexical_score",
    "dense_score",
    "term_overlap_ratio",
    "chunk_len_norm",
    "query_len_norm",
]


def _term_overlap_ratio(query: str, text: str) -> float:
    q_terms = set(query.lower().split())
    if not q_terms:
        return 0.0
    d_terms = set(text.lower().split())
    return len(q_terms & d_terms) / len(q_terms)


def build_features(query: str, candidate_text: str, lexical_score: float,
                    dense_score: float, hybrid_score: float) -> list[float]:
    return [
        hybrid_score,
        lexical_score,
        dense_score,
        _term_overlap_ratio(query, candidate_text),
        min(len(candidate_text.split()) / 200.0, 1.0),
        min(len(query.split()) / 20.0, 1.0),
    ]


def features_for_candidates(query: str, candidates: list[RetrievedChunk]) -> np.ndarray:
    return np.array([
        build_features(query, c.text, c.lexical_score, c.dense_score, c.hybrid_score)
        for c in candidates
    ])


@dataclass
class WeakExample:
    query: str
    text: str
    lexical_score: float
    dense_score: float
    hybrid_score: float
    label: int
    group: int


def generate_ict_training_data(
    chunks: list[IndexedChunk],
    embedder: Embedder,
    n_queries: int = 150,
    n_negatives: int = 4,
    seed: int = 7,
) -> list[WeakExample]:
    """Inverse Cloze Task weak supervision, as described in the module docstring."""
    rng = random.Random(seed)
    eligible = [ic for ic in chunks if len(split_sentences(ic.chunk.text)) >= 1]
    if len(eligible) < n_negatives + 1:
        return []

    all_vecs = np.vstack([ic.vector for ic in chunks])
    examples: list[WeakExample] = []

    for group_id in range(n_queries):
        source = rng.choice(eligible)
        sentences = split_sentences(source.chunk.text)
        pseudo_query = rng.choice(sentences)

        negatives_pool = [ic for ic in chunks if ic.chunk.chunk_id != source.chunk.chunk_id]
        negatives = rng.sample(negatives_pool, min(n_negatives, len(negatives_pool)))

        candidates = [source] + negatives
        labels = [1] + [0] * len(negatives)

        q_vec = embedder.embed([pseudo_query])
        cand_vecs = np.vstack([c.vector for c in candidates])
        dense_scores = cosine_sim_matrix(q_vec, cand_vecs)[0]

        for cand, label, dscore in zip(candidates, labels, dense_scores):
            lexical_score = _term_overlap_ratio(pseudo_query, cand.chunk.text)  # cheap proxy at train time
            hybrid = 0.4 * lexical_score + 0.6 * float(dscore)
            examples.append(WeakExample(
                query=pseudo_query,
                text=cand.chunk.text,
                lexical_score=lexical_score,
                dense_score=float(dscore),
                hybrid_score=hybrid,
                label=label,
                group=group_id,
            ))
    return examples


def train_reranker(examples: list[WeakExample]) -> lgb.LGBMRanker:
    if not examples:
        raise ValueError("No training examples supplied")

    X = np.array([
        build_features(e.query, e.text, e.lexical_score, e.dense_score, e.hybrid_score)
        for e in examples
    ])
    y = np.array([e.label for e in examples])

    # group sizes must reflect contiguous blocks -> examples are already grouped in order
    groups: list[int] = []
    current_group = examples[0].group
    count = 0
    for e in examples:
        if e.group != current_group:
            groups.append(count)
            current_group = e.group
            count = 0
        count += 1
    groups.append(count)

    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=60,
        num_leaves=15,
        learning_rate=0.1,
        min_child_samples=1,
        verbosity=-1,
    )
    model.fit(X, y, group=groups)
    return model


def rerank(query: str, candidates: list[RetrievedChunk], model: lgb.LGBMRanker) -> list[RetrievedChunk]:
    if not candidates:
        return []
    X = features_for_candidates(query, candidates)
    scores = model.predict(X)
    order = np.argsort(-scores)
    return [candidates[i] for i in order]
