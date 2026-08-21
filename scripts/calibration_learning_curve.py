"""
Tests whether MORE calibration data actually improves the sufficiency
gate, isolated from every other variable. A naive version of this
experiment (just raising calib_fraction) would change the test set's
composition at each step, confounding "more calibration data" with "a
different, not-directly-comparable test set" -- this holds a FIXED test
set constant across every calibration size tested, specifically to avoid
that.

Ingests the corpus and trains the reranker ONCE (neither depends on
calibration), then retrains the sufficiency gate repeatedly with
increasing amounts of calibration data -- each size is a strict prefix of
the same fixed-shuffle-order pool, so a larger calibration set is always
a superset of every smaller one, not a different random draw.

Run with: python scripts/calibration_learning_curve.py
Run against real embeddings:
         python scripts/calibration_learning_curve.py --real-embeddings
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verityrag.eval import _load, _split  # noqa: E402
from verityrag.pipeline import VerityRAGPipeline  # noqa: E402


def run_curve(embedder=None, calib_sizes=None, seed=13):
    if calib_sizes is None:
        calib_sizes = [10, 20, 40, 60, 80, 100, 120]

    corpus = _load("corpus.json")
    eval_answerable = _load("eval_answerable.json")
    eval_unanswerable = _load("eval_unanswerable.json")

    # 80/20: a large calibration POOL (120) and a FIXED test set (30) that
    # never changes across the whole experiment. calib_sizes are increasing
    # prefixes of that pool, using the same fixed shuffle order (seed),
    # so each larger set is a strict superset of every smaller one.
    calib_pool_pos, test_pos = _split(eval_answerable, calib_fraction=0.8, seed=seed)
    calib_pool_neg, test_neg = _split(eval_unanswerable, calib_fraction=0.8, seed=seed)

    pipeline = VerityRAGPipeline(embedder=embedder)
    print(f"Ingesting {len(corpus)} docs...")
    pipeline.ingest_documents(corpus)
    print("Training reranker (once, shared across every calibration size)...")
    pipeline.train_reranker(n_queries=300)

    print(f"\nFixed test set: {len(test_pos)} answerable + {len(test_neg)} unanswerable "
          f"(identical for every row below)")
    print(f"Calibration pool available: {len(calib_pool_pos)} answerable + {len(calib_pool_neg)} unanswerable\n")

    print(f"{'calib_size':>12} {'coverage':>10} {'guard':>8}")
    print("-" * 34)
    results = []
    seen_sizes = set()
    for size in calib_sizes:
        size = min(size, len(calib_pool_pos), len(calib_pool_neg))
        if size < 4 or size in seen_sizes:
            continue  # train_sufficiency_gate needs >= 8 examples total (4 pos + 4 neg minimum)
        seen_sizes.add(size)
        calib_pos = calib_pool_pos[:size]
        calib_neg = calib_pool_neg[:size]

        pipeline.calibrate_sufficiency(
            answerable_questions=[q["question"] for q in calib_pos],
            unanswerable_questions=[q["question"] for q in calib_neg],
        )

        attempted = sum(1 for item in test_pos if not pipeline.query(item["question"]).abstained)
        correctly_abstained = sum(1 for item in test_neg if pipeline.query(item["question"]).abstained)

        coverage = round(attempted / len(test_pos), 4)
        guard = round(correctly_abstained / len(test_neg), 4)
        print(f"{size:>12} {coverage:>10} {guard:>8}")
        results.append({"calib_size": size, "coverage": coverage, "guard": guard})

    return {
        "embedder": type(pipeline.embedder).__name__,
        "fixed_test_set_size": {"answerable": len(test_pos), "unanswerable": len(test_neg)},
        "curve": results,
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

    result = run_curve(embedder=chosen_embedder)

    out_path = Path(__file__).resolve().parents[1] / "calibration_learning_curve.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull detail written to {out_path}")