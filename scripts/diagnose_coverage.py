"""
Diagnoses wrongly-abstained answerable questions: for each one, was the
correct source document even retrieved, or was it retrieved but rejected by
the sufficiency gate (or, less likely now, the post-generation grounding
check)? These need completely different fixes -- a retrieval miss means the
embedding/retrieval layer itself failed to surface the right passage; a gate
rejection means retrieval worked but the confidence classifier didn't trust
it. "Look at the wrongly-abstained questions" isn't specific enough to act
on without this distinction.

Uses the exact same data split as eval.py (same seed, same fraction), so
the wrongly-abstained set here is identical to whatever eval.py reported --
this is a diagnostic on top of that run, not a different measurement.

Run with: python scripts/diagnose_coverage.py
Run against real embeddings (the actual point of this script):
         python scripts/diagnose_coverage.py --real-embeddings
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verityrag.eval import _load, _split  # noqa: E402
from verityrag.pipeline import VerityRAGPipeline  # noqa: E402
from verityrag.retrieval import hybrid_retrieve  # noqa: E402
from verityrag.sufficiency import FEATURE_NAMES, extract_features  # noqa: E402


def diagnose(embedder=None, top_k: int = 6):
    corpus = _load("corpus.json")
    eval_answerable = _load("eval_answerable.json")
    eval_unanswerable = _load("eval_unanswerable.json")

    calib_pos, test_pos = _split(eval_answerable, calib_fraction=0.5, seed=13)
    calib_neg, _test_neg = _split(eval_unanswerable, calib_fraction=0.5, seed=13)

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
    feature_importances = None
    if hasattr(gate, "model") and hasattr(gate.model, "feature_importances_"):
        feature_importances = dict(zip(FEATURE_NAMES, [int(x) for x in gate.model.feature_importances_]))
        print("\nTrained gate's actual feature importances (higher = model relies on it more):")
        for name, imp in sorted(feature_importances.items(), key=lambda x: -x[1]):
            print(f"  {name:28s} {imp}")
        print("(Ask the model directly instead of guessing from correlations --")
        print(" this is what actually drives every decision below, not a proxy for it.)\n")

    retrieval_miss = []
    gate_or_agent_rejected = []
    correctly_answered_features = []

    print(f"\nChecking {len(test_pos)} test questions...\n")
    for item in test_pos:
        result = pipeline.query(item["question"])

        # Capture features for EVERY question, not just failures -- without
        # this there's no real baseline to compare a "low" feature value
        # against, just eyeballing one group in isolation.
        candidates = hybrid_retrieve(item["question"], pipeline.index, pipeline.embedder, top_k=top_k)
        features = extract_features(candidates)

        if not result.abstained:
            correctly_answered_features.append(dict(zip(FEATURE_NAMES, [round(f, 4) for f in features])))
            continue

        # Wrongly abstained -- was the correct source document even retrieved?
        source_doc_id = item.get("source_doc_id")
        match = next((c for c in candidates if c.doc_id == source_doc_id), None)

        gate = pipeline.agent.sufficiency_gate
        gate_probability = gate.score(candidates) if hasattr(gate, "score") else None

        record = {
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "source_title": item.get("source_title"),
            "source_doc_id": source_doc_id,
            "source_doc_in_top_k": match is not None,
            "source_doc_rank": (candidates.index(match) + 1) if match else None,
            "source_doc_dense_score": round(match.dense_score, 4) if match else None,
            "top_candidate_dense_score": round(candidates[0].dense_score, 4) if candidates else None,
            "trace_actions": [s.action for s in result.trace],
            "gate_probability": round(gate_probability, 4) if gate_probability is not None else None,
            "features": dict(zip(FEATURE_NAMES, [round(f, 4) for f in features])),
        }

        if match is None:
            retrieval_miss.append(record)
        else:
            gate_or_agent_rejected.append(record)

    correctly_answered = len(correctly_answered_features)

    print("=" * 70)
    print(f"RESULTS  (embedder: {type(pipeline.embedder).__name__})")
    print("=" * 70)
    print(f"Correctly answered:              {correctly_answered}/{len(test_pos)}")
    print(f"Wrongly abstained, RETRIEVAL MISS (source doc not in top-{top_k}): "
          f"{len(retrieval_miss)}")
    print(f"Wrongly abstained, GATE/AGENT REJECTED (source doc WAS retrieved): "
          f"{len(gate_or_agent_rejected)}")

    if retrieval_miss:
        print("\n--- RETRIEVAL MISSES (embedding/retrieval layer problem) ---")
        for r in retrieval_miss:
            print(f"  Q: {r['question']}")
            print(f"     source: {r['source_title']} | top candidate dense_score: "
                  f"{r['top_candidate_dense_score']}")

    if gate_or_agent_rejected:
        print("\n--- GATE/AGENT REJECTIONS (confidence calibration problem) ---")
        for r in gate_or_agent_rejected:
            print(f"  Q: {r['question']}")
            print(f"     source: {r['source_title']} | rank {r['source_doc_rank']} "
                  f"| dense_score {r['source_doc_dense_score']} | trace: {r['trace_actions']}")
            print(f"     gate_probability: {r['gate_probability']}  (>= 0.5 needed to pass)")
            print(f"     features: {r['features']}")

        print("\n--- FEATURE COMPARISON: rejected cases vs. correctly-answered baseline ---")
        print("Computed directly from this same run's data, not a stale reference from")
        print("an earlier session -- mean, median, and range for both groups side by side.")
        for name in FEATURE_NAMES:
            rejected_vals = sorted(r["features"][name] for r in gate_or_agent_rejected)
            correct_vals = sorted(f[name] for f in correctly_answered_features)
            r_mean = round(sum(rejected_vals) / len(rejected_vals), 4)
            c_mean = round(sum(correct_vals) / len(correct_vals), 4) if correct_vals else None
            r_med = rejected_vals[len(rejected_vals) // 2]
            c_med = correct_vals[len(correct_vals) // 2] if correct_vals else None
            print(f"\n  {name}:")
            print(f"    rejected  (n={len(rejected_vals):>2}): mean={r_mean:>8}  median={r_med:>8}  "
                  f"range=[{rejected_vals[0]}, {rejected_vals[-1]}]")
            if correct_vals:
                print(f"    correct   (n={len(correct_vals):>2}): mean={c_mean:>8}  median={c_med:>8}  "
                      f"range=[{correct_vals[0]}, {correct_vals[-1]}]")
        print("\nA feature genuinely explains the rejections only if its rejected-range and")
        print("correct-range barely overlap. If the ranges overlap heavily despite a mean")
        print("difference, the two groups aren't cleanly separated by that feature alone --")
        print("same lesson as the original HashingEmbedder POS/NEG threshold problem.")

    return {
        "feature_importances": feature_importances,
        "correctly_answered": correctly_answered,
        "total": len(test_pos),
        "correctly_answered_features": correctly_answered_features,
        "retrieval_miss": retrieval_miss,
        "gate_or_agent_rejected": gate_or_agent_rejected,
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

    out_path = Path(__file__).resolve().parents[1] / "coverage_diagnosis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull detail written to {out_path}")