"""
Tests for usage_tracker.py. Mocked throughout -- never makes a real
GitHub API call or requires a real token, same principle as
test_llm_providers.py/test_elasticsearch_index.py mocking their
respective external services.

_commit_to_github (the actual HTTP logic) is tested directly and
synchronously -- it takes no Streamlit dependency at all, by design, so
these tests don't need to mock streamlit.secrets. log_input (the public,
thread-spawning wrapper) needs different tests entirely: asserting on a
mock immediately after calling it would race against the background
thread that may not have run yet. Those tests use a threading.Event to
deterministically observe "did the background work start/finish",
rather than assuming any particular timing.
"""
import base64
import json
import threading
import time
from unittest.mock import MagicMock, patch

from verityrag.usage_tracker import _commit_to_github, log_input


def _mock_secrets(token="fake-token-value"):
    """Mimics streamlit.secrets' dict-like .get() interface -- only
    needed for testing log_input's own secret-reading step, since
    _commit_to_github takes the token as a plain argument."""
    mock_st = MagicMock()
    mock_st.secrets.get.side_effect = lambda key: {"GITHUB_TOKEN": token}.get(key)
    return mock_st


# ---- _commit_to_github: the actual HTTP logic, tested synchronously ----

def test_commit_makes_correct_api_call():
    mock_response = MagicMock(status_code=201)

    with patch("requests.put", return_value=mock_response) as mock_put:
        result = _commit_to_github(
            "sk-fake-real-looking-token", "question",
            {"question": "Where was Tesla born?", "abstained": False},
        )

    assert result is True
    mock_put.assert_called_once()
    call_args = mock_put.call_args

    url = call_args.args[0]  # passed positionally in the actual module code
    assert url.startswith("https://api.github.com/repos/mugil0095/tracker-data/contents/verityrag/question_")

    headers = call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer sk-fake-real-looking-token"
    assert headers["Accept"] == "application/vnd.github+json"

    body = call_args.kwargs["json"]
    assert "message" in body
    decoded_content = json.loads(base64.b64decode(body["content"]))
    assert decoded_content["type"] == "question"
    assert decoded_content["question"] == "Where was Tesla born?"
    assert decoded_content["abstained"] is False
    assert "logged_at_utc" in decoded_content


def test_commit_returns_false_on_non_success_status_without_raising():
    mock_response = MagicMock(status_code=401, text="Bad credentials")

    with patch("requests.put", return_value=mock_response):
        result = _commit_to_github("fake-token", "question", {"question": "test?"})

    assert result is False  # must not raise even on an auth failure


def test_commit_returns_false_on_network_exception_without_raising():
    import requests

    with patch("requests.put", side_effect=requests.ConnectionError("simulated network failure")):
        result = _commit_to_github("fake-token", "question", {"question": "test?"})

    assert result is False  # a real network failure must never propagate to the caller


def test_commit_document_type_produces_a_distinct_filename_prefix():
    mock_response = MagicMock(status_code=200)

    with patch("requests.put", return_value=mock_response) as mock_put:
        _commit_to_github("fake-token", "document", {"title": "Test Doc", "text": "Some content."})

    url = mock_put.call_args.args[0]
    assert "/verityrag/document_" in url


def test_commit_each_call_gets_a_unique_filename():
    """Two separate submissions must never collide -- this is the actual
    point of the random suffix, not just the timestamp, since two
    genuinely simultaneous visitors could hit the same second."""
    mock_response = MagicMock(status_code=200)

    urls = []
    with patch("requests.put", return_value=mock_response) as mock_put:
        _commit_to_github("fake-token", "question", {"question": "first?"})
        urls.append(mock_put.call_args.args[0])
        _commit_to_github("fake-token", "question", {"question": "second?"})
        urls.append(mock_put.call_args.args[0])

    assert urls[0] != urls[1]


# ---- log_input: the public, thread-spawning wrapper ----

def test_log_input_does_not_spawn_a_thread_when_token_not_configured():
    mock_st = _mock_secrets(token=None)
    with patch.dict("sys.modules", {"streamlit": mock_st}):
        with patch("verityrag.usage_tracker._commit_to_github") as mock_commit:
            log_input("question", {"question": "test?"})
            time.sleep(0.2)  # give any (incorrectly-)spawned thread a chance to run

    mock_commit.assert_not_called()


def test_log_input_returns_immediately_even_if_the_commit_is_slow():
    """The actual point of this whole design: a slow or hanging GitHub
    call must never delay the visitor's request. Uses a real
    threading.Event rather than a fixed sleep-and-hope, so this is
    deterministic regardless of how fast or slow the test machine is."""
    mock_st = _mock_secrets(token="fake-token")
    commit_started = threading.Event()

    def slow_commit(token, input_type, data):
        commit_started.set()
        time.sleep(0.5)  # simulate a slow network call

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        with patch("verityrag.usage_tracker._commit_to_github", side_effect=slow_commit):
            start = time.monotonic()
            log_input("question", {"question": "test?"})
            elapsed = time.monotonic() - start

    assert elapsed < 0.1, f"log_input blocked for {elapsed:.3f}s -- should return near-instantly"
    assert commit_started.wait(timeout=1.0), "background thread never actually started the commit"


def test_log_input_passes_the_correct_arguments_to_the_background_commit():
    mock_st = _mock_secrets(token="fake-token-value")
    call_captured = threading.Event()
    captured_args = {}

    def capture_commit(token, input_type, data):
        captured_args["token"] = token
        captured_args["input_type"] = input_type
        captured_args["data"] = data
        call_captured.set()

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        with patch("verityrag.usage_tracker._commit_to_github", side_effect=capture_commit):
            log_input("document", {"title": "T", "text": "some text"})

    assert call_captured.wait(timeout=1.0)
    assert captured_args["token"] == "fake-token-value"
    assert captured_args["input_type"] == "document"
    assert captured_args["data"] == {"title": "T", "text": "some text"}