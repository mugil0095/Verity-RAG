"""
Eval-level regression tests: run the pipeline against small REAL slices of
the SQuAD-derived corpus/eval sets (not the full 620-doc corpus, to keep CI
fast) and assert on aggregate behavior. These exist to catch systemic
regressions (like the reformulation-drift bug -- see
test_agent_controller.py::test_reformulation_does_not_fire_when_hop0_confidently_rejected)
that per-unit-test assertions on tiny hand-built corpora won't reliably catch.

Skipped automatically if data/corpus.json hasn't been built yet (CI should
run data/build_corpus.py first -- see .github/workflows/ci.yml).
"""
import json
from pathlib import Path

import pytest

from verityrag.pipeline import VerityRAGPipeline

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "corpus.json").exists(),
    reason="data/corpus.json not built -- run `python data/build_corpus.py` first",
)


@pytest.fixture(scope="module")
def calibrated_pipeline():
    with open(DATA_DIR / "corpus.json") as f:
        corpus = json.load(f)[:250]  # small real slice -> fast CI
    with open(DATA_DIR / "eval_answerable.json") as f:
        answerable = json.load(f)[:60]
    with open(DATA_DIR / "eval_unanswerable.json") as f:
        unanswerable = json.load(f)[:60]

    pipeline = VerityRAGPipeline()
    pipeline.ingest_documents(corpus)
    pipeline.train_reranker(n_queries=150)

    calib_pos, test_pos = answerable[:30], answerable[30:]
    calib_neg, test_neg = unanswerable[:30], unanswerable[30:]
    pipeline.calibrate_sufficiency(
        answerable_questions=[q["question"] for q in calib_pos],
        unanswerable_questions=[q["question"] for q in calib_neg],
    )
    return pipeline, test_pos, test_neg


def test_hallucination_guard_rate_above_floor(calibrated_pipeline):
    pipeline, _test_pos, test_neg = calibrated_pipeline
    correctly_abstained = sum(1 for item in test_neg if pipeline.query(item["question"]).abstained)
    rate = correctly_abstained / len(test_neg)
    # Regression floor for the reformulation-drift bug. Measured directly at
    # this exact slice size: 0.0 with the bug (unconditional reformulation)
    # vs 0.4 with the fix (confidence-gated reformulation) -- see
    # test_agent_controller.py::test_reformulation_does_not_fire_when_hop0_confidently_rejected.
    # 0.2 sits cleanly between them: high enough to catch the bug returning,
    # low enough to tolerate normal run-to-run calibration variance at this
    # reduced (fast-CI) scale -- the full 620-doc eval (eval.py) is the
    # authoritative performance number (0.73 guard / 0.96 coverage), not this.
    assert rate >= 0.2, f"hallucination guard rate {rate:.2f} regressed below floor"


def test_coverage_rate_above_floor(calibrated_pipeline):
    pipeline, test_pos, _test_neg = calibrated_pipeline
    attempted = sum(1 for item in test_pos if not pipeline.query(item["question"]).abstained)
    rate = attempted / len(test_pos)
    assert rate >= 0.4, f"coverage rate {rate:.2f} regressed below floor"
