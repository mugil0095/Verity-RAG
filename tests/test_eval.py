"""
Tests for eval.py's own mechanics -- distinct from test_eval_regression.py,
which checks specific coverage/guard regression floors using its own
fixture, not run_eval() directly.
"""
import json
from pathlib import Path

import pytest

from verityrag.eval import run_eval

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def small_corpus_available():
    """run_eval() always loads the full corpus/eval files internally (see
    eval.py's _load calls) -- there's no way to inject a smaller corpus
    without changing run_eval's signature, so these tests accept the real
    full-corpus ingest cost. Kept to a single shared check rather than
    re-verifying data files exist in every test."""
    assert (DATA_DIR / "corpus.json").exists()
    assert (DATA_DIR / "eval_answerable.json").exists()
    assert (DATA_DIR / "eval_unanswerable.json").exists()


def test_max_test_questions_caps_test_set_size(small_corpus_available):
    report = run_eval(max_test_questions=3, verbose=False)
    assert report["test_set"] == {"answerable": 3, "unanswerable": 3}


def test_max_test_questions_reports_partial_sample_honestly(small_corpus_available):
    report = run_eval(max_test_questions=3, verbose=False)
    assert report["partial_sample"] is not None
    assert report["partial_sample"]["full_test_set_size"]["answerable"] == 75
    assert report["partial_sample"]["full_test_set_size"]["unanswerable"] == 75
    assert "SMALL SAMPLE" in report["partial_sample"]["note"]


def test_no_max_test_questions_means_no_partial_sample_field(small_corpus_available):
    """Default (unbounded) run must not claim to be partial -- this is the
    backward-compatibility guarantee for every existing eval number already
    documented in README/ROADMAP, which were all measured before this
    parameter existed."""
    report = run_eval(verbose=False)
    assert report["partial_sample"] is None
    assert report["test_set"] == {"answerable": 75, "unanswerable": 75}


def test_max_test_questions_metrics_computed_against_the_smaller_sample(small_corpus_available):
    """coverage_rate/hallucination_guard_rate must be fractions of the
    SAMPLE size, not silently still dividing by the full 75 -- a report
    that said e.g. '2/75 attempted' when only 3 questions were even asked
    would be a real, misleading bug, not just an odd number."""
    report = run_eval(max_test_questions=3, verbose=False)
    assert report["attempted"] <= 3
    assert report["coverage_rate"] == round(report["attempted"] / 3, 3)