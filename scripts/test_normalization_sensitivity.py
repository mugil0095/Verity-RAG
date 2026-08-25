"""
Tests the one named-but-untested hypothesis left on the hybrid-ranking
investigation (see ROADMAP.md): hybrid_retrieve() min-max normalizes
lexical and dense scores over the ENTIRE index snapshot, not just the
top-k candidates that end up mattering (confirmed directly in
retrieval.py -- lexical_norm/dense_norm are computed on the full
snapshot). The hypothesis is that this makes the normalized scale
sensitive to per-query outliers: if some chunk elsewhere in the corpus
(not among the real top candidates) happens to have an unusually high
raw score for a given query, it could compress the scale everything
else gets measured against, distorting the ranking among the candidates
that actually compete for rank 1.

This is directly testable: for every case where the wrong document wins
rank 1 (gold document retrieved but not first), check whether the
index-wide max lexical/dense score is coming from WITHIN the actual
top-6 candidates, or from some OTHER chunk entirely. If it's usually
from outside the top-6, that's real evidence for outlier-driven
compression. If it's usually from within the top-6 anyway, the
hypothesis doesn't hold -- the competition is genuinely among the real
candidates, not distorted by some unrelated outlier elsewhere.

Run with: python scripts/test_normalization_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verityrag.eval import _load  # noqa: E402
from verityrag.pipeline import VerityRAGPipeline  # noqa: E402
from verityrag.indexing import LiveIndex  # noqa: E402
from verityrag.embedding import cosine_sim_matrix  # noqa: E402


def diagnose(embedder=None, top_k: int = 6):
    corpus = _load("corpus.json")
    eval_answerable = _load("eval_answerable.json")

    pipeline = VerityRAGPipeline(embedder=embedder)
    print(f"Ingesting {len(corpus)} docs...")
    pipeline.ingest_documents(corpus)

    index: LiveIndex = pipeline.index
    snapshot, doc_vecs = index.snapshot_with_matrix()
    chunk_ids = [ic.chunk.chunk_id for ic in snapshot]

    wrong_rank1_cases = []
    print(f"\nChecking all {len(eval_answerable)} answerable questions for wrong-rank-1 cases...\n")
    for item in eval_answerable:
        gold_doc_id = item.get("source_doc_id")
        if not gold_doc_id:
            continue

        query = item["question"]
        lexical_scores = index.lexical_index.scores(query, chunk_ids)
        query_vec = pipeline.embedder.embed([query])
        dense_scores = cosine_sim_matrix(query_vec, doc_vecs)[0]

        # Same ordering hybrid_retrieve() itself uses, so "top-6" here
        # matches exactly what a real query would surface.
        lexical_norm = (lexical_scores - lexical_scores.min()) / max(
            lexical_scores.max() - lexical_scores.min(), 1e-9)
        dense_norm = (dense_scores - dense_scores.min()) / max(
            dense_scores.max() - dense_scores.min(), 1e-9)
        hybrid = 0.4 * lexical_norm + 0.6 * dense_norm
        top6_idx = np.argsort(-hybrid)[:top_k]

        winner_idx = top6_idx[0]
        winner_doc_id = snapshot[winner_idx].chunk.doc_id
        gold_in_top6 = any(snapshot[i].chunk.doc_id == gold_doc_id for i in top6_idx)

        if winner_doc_id == gold_doc_id or not gold_in_top6:
            continue  # only care about wrong-rank-1-but-gold-in-top6 cases

        # THE ACTUAL TEST: is the index-wide max lexical/dense score
        # coming from a chunk that's actually in the top-6, or from
        # somewhere else entirely?
        max_lexical_idx = int(np.argmax(lexical_scores))
        max_dense_idx = int(np.argmax(dense_scores))
        lexical_max_from_outside_top6 = max_lexical_idx not in top6_idx
        dense_max_from_outside_top6 = max_dense_idx not in top6_idx

        wrong_rank1_cases.append({
            "question": query,
            "lexical_max_from_outside_top6": lexical_max_from_outside_top6,
            "dense_max_from_outside_top6": dense_max_from_outside_top6,
        })

    n = len(wrong_rank1_cases)
    lexical_outside_count = sum(1 for c in wrong_rank1_cases if c["lexical_max_from_outside_top6"])
    dense_outside_count = sum(1 for c in wrong_rank1_cases if c["dense_max_from_outside_top6"])

    print("=" * 70)
    print(f"RESULTS  (n={n} wrong-rank-1-but-gold-in-top6 cases)")
    print("=" * 70)
    print(f"Lexical-score max came from OUTSIDE the top-6: {lexical_outside_count}/{n} "
          f"({lexical_outside_count/n*100:.0f}%)" if n else "n/a")
    print(f"Dense-score max came from OUTSIDE the top-6:   {dense_outside_count}/{n} "
          f"({dense_outside_count/n*100:.0f}%)" if n else "n/a")
    print()
    print("If these percentages are LOW, the hypothesis doesn't hold: the")
    print("normalization scale is being set by the real candidates themselves,")
    print("not distorted by some unrelated outlier elsewhere in the corpus.")
    print("If HIGH, that's real evidence for outlier-driven compression.")

    return {
        "n_wrong_rank1_cases": n,
        "lexical_outside_count": lexical_outside_count,
        "dense_outside_count": dense_outside_count,
        "cases": wrong_rank1_cases,
    }


if __name__ == "__main__":
    import json

    result = diagnose()
    out_path = Path(__file__).resolve().parents[1] / "normalization_sensitivity_check.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull detail written to {out_path}")