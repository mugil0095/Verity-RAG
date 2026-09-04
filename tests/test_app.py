"""
Tests for app.py using Streamlit's official AppTest framework (simulates a
real browser session against the script, without needing a running server).

Skipped if data/corpus.json hasn't been built yet -- same convention as
test_eval_regression.py.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "corpus.json").exists(),
    reason="data/corpus.json not built -- run `python data/build_corpus.py` first",
)


@pytest.fixture(autouse=True)
def _clear_shared_pipeline_cache():
    """app.py's main pipeline is a deliberate, shared @st.cache_resource
    singleton -- one instance for every real browser session against one
    running server, which is exactly the point (see app.py's own module
    docstring). But that means it's also scoped to the PYTHON PROCESS, not
    to individual AppTest instances -- confirmed directly: two separate,
    independently-constructed AppTest.from_file() calls in the same pytest
    process return the literal same pipeline object (same id()), so one
    test's ingested documents leak into the next test's "fresh" instance.
    Without this, tests only happened to pass because of file ordering
    (the one test asserting an empty index runs first, before anything
    could pollute it) -- not because isolation genuinely held. Clearing
    the cache before each test restores real per-test isolation without
    touching the sharing behavior the real app actually wants."""
    st.cache_resource.clear()
    yield


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
    assert at.session_state.pipeline._reranker_model is not None


def test_sidebar_calibration_updates_gate():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    [b for b in at.sidebar.button if b.label == "Load corpus + train reranker"][0].click().run(timeout=60)
    calib_btn = [b for b in at.sidebar.button if "Calibrate" in b.label][0]
    calib_btn.click().run(timeout=60)
    assert not at.exception
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
    """Deliberately uses the lightweight manual-document form, not the full
    620-doc "Load corpus" button -- Reset's correctness doesn't depend on
    HOW MUCH is in the index, only that it empties it, and this is the last
    test in the file: by this point it carries the accumulated memory
    pressure of every heavier test before it. A real MemoryError from the
    full corpus showed up here specifically (845 chunks x 65536-dim dense
    vectors = 422MB) even with conftest.py's gc.collect() between tests --
    that mitigation measurably helped elsewhere in this file but wasn't
    enough for the single most cumulative test. Using ~1 chunk instead of
    845 sidesteps the problem at its actual root for this specific test,
    without weakening what Reset itself is being tested against."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    at.sidebar.text_input[0].set_value("Custom Doc")
    at.sidebar.text_area[0].set_value("A real sentence about quantum computing and qubits.")
    [b for b in at.sidebar.button if "Ingest" in b.label][0].click().run(timeout=30)
    assert at.session_state.pipeline.index.size() > 0

    at.sidebar.checkbox[0].set_value(True).run(timeout=30)  # must confirm AND re-run before Reset is enabled
    reset_btn = [b for b in at.sidebar.button if "Reset" in b.label][0]
    reset_btn.click().run(timeout=30)
    assert not at.exception
    assert at.session_state.pipeline.index.size() == 0


def test_reset_button_disabled_without_confirmation():
    """The confirmation checkbox is the actual point of this change --
    make sure Reset genuinely can't fire without it, not just that the
    checkbox exists."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    reset_btn = [b for b in at.sidebar.button if "Reset" in b.label][0]
    assert reset_btn.disabled is True


def test_llm_comparison_checkbox_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    checkbox = [c for c in at.checkbox if "LLM" in c.label][0]
    assert checkbox.disabled is True


def test_llm_comparison_shows_both_results_when_enabled(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")

    mock_response = MagicMock()
    mock_response.text = "Tesla contributed major advances to AC electrical systems."
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    [b for b in at.sidebar.button if b.label == "Load corpus + train reranker"][0].click().run(timeout=60)

    checkbox = [c for c in at.checkbox if "LLM" in c.label][0]
    assert checkbox.disabled is False  # key IS configured now, unlike the previous test
    checkbox.set_value(True)
    at.text_input(key="ask_question").set_value("What is Nikola Tesla known for?")

    with patch("google.genai.Client", return_value=mock_client):
        [b for b in at.button if b.label == "Ask"][0].click().run(timeout=30)

    assert not at.exception
    subheaders = [h.value for h in at.subheader]
    assert "Extractive (default)" in subheaders
    assert "Real LLM (Gemini)" in subheaders
    mock_client.models.generate_content.assert_called_once()


def test_llm_comparison_handles_failure_gracefully(monkeypatch):
    """The extractive answer must still work even when the LLM side fails
    for any reason -- this is the actual point of the try/except in
    app.py, not just that a comparison generally works."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")

    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=30)
    [b for b in at.sidebar.button if b.label == "Load corpus + train reranker"][0].click().run(timeout=60)

    checkbox = [c for c in at.checkbox if "LLM" in c.label][0]
    checkbox.set_value(True)
    at.text_input(key="ask_question").set_value("What is Nikola Tesla known for?")

    with patch("google.genai.Client", side_effect=RuntimeError("simulated daily quota exceeded")):
        [b for b in at.button if b.label == "Ask"][0].click().run(timeout=30)

    assert not at.exception  # the failure must be caught, not crash the app
    subheaders = [h.value for h in at.subheader]
    assert "Extractive (default)" in subheaders  # still rendered despite the LLM side failing
    error_messages = [e.value for e in at.error]
    assert any("LLM comparison unavailable" in msg for msg in error_messages)