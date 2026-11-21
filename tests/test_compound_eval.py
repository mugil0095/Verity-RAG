"""
Regression tests for the multi-agent loop on the real corpus: it must beat the
single-agent loop on two-part questions and must not change a single decision
on the single-part test questions (which is what lets it stay on by default).
"""
from pathlib import Path

import pytest

from verityrag.compound_eval import build_pairs, build_pipeline, evaluate_pairs, mcnemar_exact, wilson

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "corpus.json").exists(),
    reason="data/corpus.json not built -- run `python data/build_corpus.py` first",
)


@pytest.fixture(scope="module")
def evaluated():
    pipeline, test_pos = build_pipeline()
    pairs = build_pairs(test_pos)
    return pipeline, test_pos, pairs, evaluate_pairs(pipeline, pairs)


def test_pairs_are_deterministic_and_come_from_the_test_half_only(evaluated):
    _pipeline, test_pos, pairs, _report = evaluated
    assert [p["question"] for p in pairs] == [p["question"] for p in build_pairs(test_pos)]
    test_questions = {q["question"] for q in test_pos}
    assert all(p["a"]["question"] in test_questions and p["b"]["question"] in test_questions for p in pairs)
    assert {p["template"] for p in pairs} == {"sentences", "conjunction"}


def test_multi_agent_beats_single_agent_on_two_part_questions(evaluated):
    report = evaluated[3]
    single, multi = report["single_agent"]["both_answers"], report["multi_agent"]["both_answers"]
    # Measured 11 vs 30 of 74; the floor leaves room for run-to-run calibration variance.
    assert multi >= 2 * single and multi - single >= 10
    assert report["mcnemar_both_answers"]["p_value"] < 0.01
    assert report["multi_agent"]["both_answers"] <= report["perfect_split"]["both_answers"] + 3


def test_planner_splits_most_pairs(evaluated):
    report = evaluated[3]
    assert report["split_by_planner"] >= 0.8 * report["pairs"]


def test_single_part_questions_get_identical_results_in_both_modes(evaluated):
    pipeline, test_pos, _pairs, _report = evaluated
    for item in test_pos:
        pipeline.multi_agent = False
        before = pipeline.query(item["question"])
        pipeline.multi_agent = True
        after = pipeline.query(item["question"])
        assert (after.abstained, after.abstain_reason, after.answer) == (before.abstained, before.abstain_reason, before.answer)
        assert [c.chunk_id for c in after.evidence] == [c.chunk_id for c in before.evidence]


def test_statistics_helpers():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 10) == 1.0
    assert mcnemar_exact(19, 0) == pytest.approx(2 / 2 ** 19)
    lo, hi = wilson(30, 74)
    assert lo < 30 / 74 < hi
    assert wilson(0, 0) == (0.0, 0.0)
