"""
Checks, across the full answerable question set, how often hybrid_retrieve
puts the WRONG document at rank 1 -- and when it does, whether the winner
beat the correct document on dense score, lexical score, or both. This is
a different question from "was the correct document retrieved at all"
(scripts/diagnose_coverage.py already answers that) -- this is specifically
about ranking quality among documents that DID get retrieved.

Motivated by a real, structural property of hybrid_retrieve (retrieval.py):
min-max normalization happens over the ENTIRE index snapshot, not just the
top-K. That makes it sensitive to per-query outliers -- if some unrelated
chunk has an unusually high BM25 score for a given query, it can compress
the whole scale everything else gets measured against. This checks whether
that's actually happening at any meaningful rate, rather than assuming it
from 1-2 anecdotal cases found earlier.

Run with: python scripts/diagnose_hybrid_ranking.py
Run against real embeddings:
         python scripts/diagnose_hybrid_ranking.py --real-embeddings
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verityrag.eval import _load  # noqa: E402
from verityrag.pipeline import VerityRAGPipeline  # noqa: E402
from verityrag.retrieval import hybrid_retrieve  # noqa: E402


def diagnose(embedder=None, top_k: int = 6):
    corpus = _load("corpus.json")
    eval_answerable = _load("eval_answerable.json")

    pipeline = VerityRAGPipeline(embedder=embedder)
    print(f"Ingesting {len(corpus)} docs...")
    pipeline.ingest_documents(corpus)

    correct_at_rank1 = 0
    wrong_at_rank1 = []
    not_in_top_k = 0

    print(f"\nChecking all {len(eval_answerable)} answerable questions...\n")
    for item in eval_answerable:
        gold_doc_id = item.get("source_doc_id")
        if not gold_doc_id:
            continue

        candidates = hybrid_retrieve(item["question"], pipeline.index, pipeline.embedder, top_k=top_k)
        if not candidates:
            continue

        if candidates[0].doc_id == gold_doc_id:
            correct_at_rank1 += 1
            continue

        gold_candidate = next((c for c in candidates if c.doc_id == gold_doc_id), None)
        if gold_candidate is None:
            not_in_top_k += 1
            continue

        winner = candidates[0]
        gold_rank = candidates.index(gold_candidate) + 1
        wrong_at_rank1.append({
            "question": item["question"],
            "gold_doc_id": gold_doc_id,
            "gold_rank": gold_rank,
            "gold_dense": round(gold_candidate.dense_score, 4),
            "gold_lexical": round(gold_candidate.lexical_score, 4),
            "winner_doc_id": winner.doc_id,
            "winner_dense": round(winner.dense_score, 4),
            "winner_lexical": round(winner.lexical_score, 4),
            "winner_beat_gold_on": (
                "both" if winner.dense_score > gold_candidate.dense_score and winner.lexical_score > gold_candidate.lexical_score
                else "dense_only" if winner.dense_score > gold_candidate.dense_score
                else "lexical_only" if winner.lexical_score > gold_candidate.lexical_score
                else "neither_raw_score_but_won_hybrid"
            ),
        })

    total = correct_at_rank1 + len(wrong_at_rank1) + not_in_top_k
    print("=" * 70)
    print(f"RESULTS  (embedder: {type(pipeline.embedder).__name__}, n={total})")
    print("=" * 70)
    print(f"Correct document at rank 1:              {correct_at_rank1}/{total}")
    print(f"Wrong document at rank 1 (gold in top-{top_k}): {len(wrong_at_rank1)}/{total}")
    print(f"Gold document not in top-{top_k} at all:     {not_in_top_k}/{total}")

    if wrong_at_rank1:
        from collections import Counter
        reasons = Counter(w["winner_beat_gold_on"] for w in wrong_at_rank1)
        print(f"\nWhen the wrong document won, what beat the correct one:")
        for reason, count in reasons.most_common():
            print(f"  {reason:35s} {count}")

        print(f"\n--- All {len(wrong_at_rank1)} cases ---")
        for w in wrong_at_rank1:
            print(f"  Q: {w['question']}")
            print(f"     gold (rank {w['gold_rank']}): dense={w['gold_dense']}  lexical={w['gold_lexical']}")
            print(f"     winner (rank 1):    dense={w['winner_dense']}  lexical={w['winner_lexical']}"
                  f"   [beat gold on: {w['winner_beat_gold_on']}]")

    return {
        "embedder": type(pipeline.embedder).__name__,
        "total": total,
        "correct_at_rank1": correct_at_rank1,
        "not_in_top_k": not_in_top_k,
        "wrong_at_rank1": wrong_at_rank1,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--real-embeddings", action="store_true")
    args = parser.parse_args()

    chosen_embedder = None
    if args.real_embeddings:
        from verityrag.embedding import SentenceTransformerEmbedder
        chosen_embedder = SentenceTransformerEmbedder()

    result = diagnose(embedder=chosen_embedder)

    out_path = Path(__file__).resolve().parents[1] / "hybrid_ranking_diagnosis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull detail written to {out_path}")