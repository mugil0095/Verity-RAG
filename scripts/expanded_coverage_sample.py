"""
The sufficiency-gate/retrieval-miss investigation has repeatedly hit the
same wall: only 5 failures at the current 93.3% coverage baseline is too
few for correlational analysis to find a real pattern (see ROADMAP.md).
The only genuine way past that is more labeled data, not a different
split ratio or another feature -- this generates a larger SUPPLEMENTARY
answerable-question sample (same 12 topics, same methodology as
data/build_corpus.py, just a higher per-paragraph cap) to get a bigger
failure sample to actually analyze.

Deliberately does NOT touch data/eval_answerable.json or corpus.json --
every existing baseline number in this project depends on that exact
150-question set staying fixed. This writes its own separate file and
uses the existing corpus (documents don't need to change, only which
questions get asked against them).

Run with: python scripts/expanded_coverage_sample.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build_expanded_sample(qas_per_paragraph: int = 5, target_size: int = 400, seed: int = 42):
    """Mirrors build_corpus.py's exact question-selection logic, just with
    a higher per-paragraph cap and a different target size. Uses the SAME
    seed and the SAME 12 included topics, so this is an apples-to-apples
    extension of the existing sample, not a differently-sourced one."""
    squad_path = DATA_DIR / "squad_dev.json"
    with open(squad_path) as f:
        squad = json.load(f)

    included_topics = squad["data"][:12]  # matches build_corpus.py exactly

    eval_answerable = []
    doc_id = 0
    for topic in included_topics:
        title = topic["title"]
        for para in topic["paragraphs"]:
            for qa in para["qas"][:qas_per_paragraph]:
                if qa.get("is_impossible") or not qa["answers"]:
                    continue
                eval_answerable.append({
                    "question": qa["question"],
                    "gold_answer": qa["answers"][0]["text"],
                    "source_doc_id": f"d{doc_id}",
                    "source_title": title,
                })
            doc_id += 1

    random.Random(seed).shuffle(eval_answerable)
    return eval_answerable[:target_size]


def diagnose(embedder=None):
    from verityrag.eval import _load
    from verityrag.pipeline import VerityRAGPipeline
    from verityrag.retrieval import hybrid_retrieve
    from verityrag.sufficiency import FEATURE_NAMES, extract_features

    corpus = _load("corpus.json")
    expanded = build_expanded_sample()
    print(f"Expanded answerable sample: {len(expanded)} questions "
          f"(vs. the existing 150-question baseline)")

    pipeline = VerityRAGPipeline(embedder=embedder)
    print(f"Ingesting {len(corpus)} docs...")
    pipeline.ingest_documents(corpus)
    print("Training reranker...")
    pipeline.train_reranker(n_queries=300)

    # Calibrate using the EXISTING 75/75 calibration split, unchanged --
    # only the TEST question set is expanded here, so the gate itself is
    # trained exactly as it always is elsewhere in this project.
    eval_answerable = _load("eval_answerable.json")
    eval_unanswerable = _load("eval_unanswerable.json")
    from verityrag.eval import _split
    calib_pos, _ = _split(eval_answerable, calib_fraction=0.5, seed=13)
    calib_neg, _ = _split(eval_unanswerable, calib_fraction=0.5, seed=13)
    print("Calibrating sufficiency gate (existing 75/75 split, unchanged)...")
    pipeline.calibrate_sufficiency(
        answerable_questions=[q["question"] for q in calib_pos],
        unanswerable_questions=[q["question"] for q in calib_neg],
    )

    attempted = 0
    wrongly_abstained_features = []
    correct_features = []
    for item in expanded:
        result = pipeline.query(item["question"])
        # Recompute the SAME features the gate itself saw at hop-0, for
        # comparison -- mirrors diagnose_coverage.py exactly, since this
        # is the same kind of before/after feature comparison, just on a
        # bigger failure sample.
        candidates = hybrid_retrieve(item["question"], pipeline.index, pipeline.embedder, top_k=6)
        features = extract_features(candidates)
        if result.abstained:
            wrongly_abstained_features.append(features)
        else:
            attempted += 1
            correct_features.append(features)

    coverage = attempted / len(expanded)
    print(f"\nCoverage on expanded sample: {coverage:.3f} ({attempted}/{len(expanded)})")
    print(f"Wrongly abstained: {len(wrongly_abstained_features)}  "
          f"<- this is the failure sample we actually care about")

    if len(wrongly_abstained_features) >= 5:
        print(f"\n--- FEATURE COMPARISON: rejected (n={len(wrongly_abstained_features)}) "
              f"vs. correct (n={len(correct_features)}) ---")
        import numpy as np
        from scipy import stats
        rejected_arr = np.array(wrongly_abstained_features)
        correct_arr = np.array(correct_features)
        for i, name in enumerate(FEATURE_NAMES):
            r = rejected_arr[:, i]
            c = correct_arr[:, i]
            overlap = not (r.max() < c.min() or c.max() < r.min())
            # Mann-Whitney U rather than a t-test -- doesn't assume the
            # feature values are normally distributed, which these
            # clearly aren't (dense/lexical scores are bounded, skewed).
            u_stat, p_value = stats.mannwhitneyu(r, c, alternative="two-sided")
            significant = p_value < 0.01
            print(f"  {name}:")
            print(f"    rejected  (n={len(r):>3}): mean={r.mean():.4f}  "
                  f"range=[{r.min():.4f}, {r.max():.4f}]")
            print(f"    correct   (n={len(c):>3}): mean={c.mean():.4f}  "
                  f"range=[{c.min():.4f}, {c.max():.4f}]")
            print(f"    ranges overlap: {overlap}   "
                  f"Mann-Whitney p={p_value:.6f}  (significant at p<0.01: {significant})")
    else:
        print(f"\nStill fewer than 5 failures ({len(wrongly_abstained_features)}) -- "
              f"even this larger sample didn't surface enough to analyze. "
              f"Coverage may simply be genuinely high at this corpus size.")

    return {
        "expanded_sample_size": len(expanded),
        "coverage": coverage,
        "n_wrongly_abstained": len(wrongly_abstained_features),
        "wrongly_abstained_features": [list(f) for f in wrongly_abstained_features],
        "correct_features": [list(f) for f in correct_features],
        "feature_names": FEATURE_NAMES,
    }


if __name__ == "__main__":
    result = diagnose()
    out_path = Path(__file__).resolve().parents[1] / "expanded_coverage_sample.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull detail written to {out_path}")