"""
Sweeps the sufficiency gate's decision_threshold to see whether a different
cutoff would improve the coverage/hallucination-guard trade-off, using the
ALREADY-trained/calibrated classifier -- no retraining needed, so this is
cheap: one pass through the test set, not one full pipeline run per
threshold tried.

IMPORTANT APPROXIMATION, not a perfect replica of full pipeline behavior:
this scores each question's hop-0 probability directly (hybrid_retrieve +
gate.score()) and re-thresholds that SAME number at several candidate
cutoffs in Python. The real agent (agent.py) can reformulate and retry
across up to 2 hops when hop-0 is borderline -- a question this script
marks "fails at threshold 0.3" might actually get rescued by reformulation
in the real pipeline, and vice versa. This is a fast, directional signal
for whether threshold-tuning is even worth pursuing further -- confirm any
promising threshold with a real eval.py run (or a small change to
CalibratedSufficiencyGate.decision_threshold + a real pipeline run) before
trusting it, same discipline as everything else in this investigation.

Run with: python scripts/sweep_threshold.py
Run against real embeddings (the actual point of this script):
         python scripts/sweep_threshold.py --real-embeddings
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verityrag.eval import _load, _split  # noqa: E402
from verityrag.pipeline import VerityRAGPipeline  # noqa: E402
from verityrag.retrieval import hybrid_retrieve  # noqa: E402


def sweep(embedder=None, top_k: int = 6, thresholds=None):
    if thresholds is None:
        thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    corpus = _load("corpus.json")
    eval_answerable = _load("eval_answerable.json")
    eval_unanswerable = _load("eval_unanswerable.json")

    calib_pos, test_pos = _split(eval_answerable, calib_fraction=0.5, seed=13)
    calib_neg, test_neg = _split(eval_unanswerable, calib_fraction=0.5, seed=13)

    pipeline = VerityRAGPipeline(embedder=embedder)
    print(f"Ingesting {len(corpus)} docs...")
    pipeline.ingest_documents(corpus)
    print("Training reranker...")
    pipeline.train_reranker(n_queries=300)
    print("Calibrating sufficiency gate...")
    pipeline.calibrate_sufficiency(
        answerable_questions=[q["question"] for q in calib_pos],
        unanswerable_questions=[q["question"] for q in calib_neg],
    )

    gate = pipeline.agent.sufficiency_gate
    print(f"Current decision_threshold: {gate.decision_threshold}")

    print(f"\nScoring {len(test_pos)} answerable + {len(test_neg)} unanswerable "
          f"test questions (hop-0 only, one pass)...")

    pos_probs = []
    for item in test_pos:
        candidates = hybrid_retrieve(item["question"], pipeline.index, pipeline.embedder, top_k=top_k)
        pos_probs.append(gate.score(candidates) if candidates else 0.0)

    neg_probs = []
    for item in test_neg:
        candidates = hybrid_retrieve(item["question"], pipeline.index, pipeline.embedder, top_k=top_k)
        neg_probs.append(gate.score(candidates) if candidates else 0.0)

    print(f"\n(embedder: {type(pipeline.embedder).__name__})")
    print("APPROXIMATION -- hop-0 probability only, does not model reformulation.")
    print("See module docstring before trusting a threshold from this alone.\n")
    print(f"{'threshold':>10} {'coverage':>10} {'guard':>10}")
    print("-" * 34)
    results = []
    for t in thresholds:
        coverage = round(sum(1 for p in pos_probs if p >= t) / len(pos_probs), 4)
        guard = round(sum(1 for p in neg_probs if p < t) / len(neg_probs), 4)
        marker = "  <- current default" if abs(t - gate.decision_threshold) < 1e-9 else ""
        print(f"{t:>10.2f} {coverage:>10.3f} {guard:>10.3f}{marker}")
        results.append({"threshold": t, "coverage_approx": coverage, "guard_approx": guard})

    return {
        "embedder": type(pipeline.embedder).__name__,
        "current_decision_threshold": gate.decision_threshold,
        "pos_probs": [round(p, 4) for p in pos_probs],
        "neg_probs": [round(p, 4) for p in neg_probs],
        "sweep": results,
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

    result = sweep(embedder=chosen_embedder)

    out_path = Path(__file__).resolve().parents[1] / "threshold_sweep.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull detail written to {out_path}")