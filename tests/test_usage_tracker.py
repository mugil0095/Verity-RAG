"""
Tests for usage_tracker.py. Mocked throughout -- never makes a real
GitHub API call or requires a real token, same principle as
test_llm_providers.py/test_elasticsearch_index.py mocking their
respective external services.
"""
import base64
import json
from unittest.mock import MagicMock, patch

from verityrag.usage_tracker import log_input


def _mock_secrets(token="fake-token-value"):
    """Mimics streamlit.secrets' dict-like .get() interface."""
    mock_st = MagicMock()
    mock_st.secrets.get.side_effect = lambda key: {"GITHUB_TOKEN": token}.get(key)
    return mock_st


def test_returns_false_without_raising_when_token_not_configured():
    mock_st = _mock_secrets(token=None)
    with patch.dict("sys.modules", {"streamlit": mock_st}):
        with patch("requests.put") as mock_put:
            result = log_input("question", {"question": "test?"})

    assert result is False
    mock_put.assert_not_called()  # must not attempt a network call with no token


def test_makes_correct_api_call_when_token_is_configured():
    mock_st = _mock_secrets(token="sk-fake-real-looking-token")
    mock_response = MagicMock(status_code=201)

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        with patch("requests.put", return_value=mock_response) as mock_put:
            result = log_input("question", {"question": "Where was Tesla born?", "abstained": False})

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


def test_returns_false_on_non_success_status_without_raising():
    mock_st = _mock_secrets()
    mock_response = MagicMock(status_code=401, text="Bad credentials")

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        with patch("requests.put", return_value=mock_response):
            result = log_input("question", {"question": "test?"})

    assert result is False  # must not raise even on an auth failure


def test_returns_false_on_network_exception_without_raising():
    import requests

    mock_st = _mock_secrets()
    with patch.dict("sys.modules", {"streamlit": mock_st}):
        with patch("requests.put", side_effect=requests.ConnectionError("simulated network failure")):
            result = log_input("question", {"question": "test?"})

    assert result is False  # a real network failure must never propagate to the caller


def test_document_type_produces_a_distinct_filename_prefix():
    mock_st = _mock_secrets()
    mock_response = MagicMock(status_code=200)

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        with patch("requests.put", return_value=mock_response) as mock_put:
            log_input("document", {"title": "Test Doc", "text": "Some content."})

    url = mock_put.call_args.args[0]
    assert "/verityrag/document_" in url


def test_each_call_gets_a_unique_filename():
    """Two separate submissions must never collide -- this is the actual
    point of the random suffix, not just the timestamp, since two
    genuinely simultaneous visitors could hit the same second."""
    mock_st = _mock_secrets()
    mock_response = MagicMock(status_code=200)

    urls = []
    with patch.dict("sys.modules", {"streamlit": mock_st}):
        with patch("requests.put", return_value=mock_response) as mock_put:
            log_input("question", {"question": "first?"})
            urls.append(mock_put.call_args.args[0])
            log_input("question", {"question": "second?"})
            urls.append(mock_put.call_args.args[0])

    assert urls[0] != urls[1]