"""
Tests for app.py using Streamlit's official AppTest framework (simulates a
real browser session against the script, without needing a running server).

Skipped if data/corpus.json hasn't been built yet -- same convention as
test_eval_regression.py.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "corpus.json").exists(),
    reason="data/corpus.json not built -- run `python data/build_corpus.py` first",
)


def test_app_loads_without_exceptions():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    assert not at.exception
    assert at.session_state.pipeline.index.size() == 0


def test_sidebar_load_corpus_populates_index():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    load_btn = [b for b in at.sidebar.button if b.label == "Load corpus + train reranker"][0]
    load_btn.click().run(timeout=60)
    assert not at.exception
    assert at.session_state.pipeline.index.size() > 0
    assert at.session_state.reranker_trained is True


def test_sidebar_calibration_updates_gate():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    [b for b in at.sidebar.button if b.label == "Load corpus + train reranker"][0].click().run(timeout=60)
    calib_btn = [b for b in at.sidebar.button if "Calibrate" in b.label][0]
    calib_btn.click().run(timeout=60)
    assert not at.exception
    assert at.session_state.gate_calibrated is True
    assert at.session_state.pipeline.agent.sufficiency_gate.is_calibrated is True


def test_manual_document_ingestion_via_sidebar_form():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    at.sidebar.text_input[0].set_value("Custom Doc")
    at.sidebar.text_area[0].set_value("A real sentence about quantum computing and qubits.")
    submit = [b for b in at.sidebar.button if "Ingest" in b.label][0]
    submit.click().run(timeout=30)
    assert not at.exception
    assert at.session_state.pipeline.index.size() >= 1


def test_ask_tab_returns_an_answer_after_loading_corpus():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    [b for b in at.sidebar.button if b.label == "Load corpus + train reranker"][0].click().run(timeout=60)

    at.text_input(key="ask_question").input("What is Nikola Tesla known for?").run(timeout=30)
    ask_btn = [b for b in at.button if b.label == "Ask"][0]
    ask_btn.click().run(timeout=30)
    assert not at.exception
    # either a real success (answered) or a warning (abstained) -- both are
    # valid outcomes depending on gate calibration state, but one must appear
    assert len(at.success) > 0 or len(at.warning) > 0


def test_demo_tab_shows_correct_before_after_and_abstention_on_unanswerable():
    """The headline end-to-end scenario: refuse before ingestion, answer
    after streaming, still refuse a genuinely out-of-domain question."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)

    setup_btn = [b for b in at.button if b.label == "Set up demo"][0]
    setup_btn.click().run(timeout=90)
    assert not at.exception
    assert at.session_state.demo_pipeline.index.size() > 0

    run_btn = [b for b in at.button if "Run the live streaming demo" in b.label][0]
    run_btn.click().run(timeout=60)
    assert not at.exception

    assert at.session_state.demo_before.abstained is True
    assert at.session_state.demo_after.abstained is False
    assert at.session_state.demo_after.answer
    assert at.session_state.demo_unanswerable.abstained is True


def test_reset_button_clears_the_index():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    [b for b in at.sidebar.button if b.label == "Load corpus + train reranker"][0].click().run(timeout=60)
    assert at.session_state.pipeline.index.size() > 0

    reset_btn = [b for b in at.sidebar.button if "Reset" in b.label][0]
    reset_btn.click().run(timeout=30)
    assert not at.exception
    assert at.session_state.pipeline.index.size() == 0
