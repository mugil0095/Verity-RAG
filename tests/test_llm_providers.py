"""
Tests for llm_providers.py: anthropic_complete_fn and gemini_complete_fn.
Mocked throughout -- never makes a real API call or requires a real API
key, same principle as test_sentence_embedder.py mocking the
sentence-transformers model instead of downloading a real one for
fast/offline/deterministic tests. test_sentence_embedder_live.py is the
pattern for a SEPARATE, explicitly-opt-in live test if one is ever wanted
here too -- not added in this pass: Anthropic's would need a real paid
call every run (a real cost this project's existing live tests don't
have), and Gemini's free tier, while real, is rate-limited enough that a
live test could flake in CI for reasons unrelated to code correctness.
"""
from unittest.mock import MagicMock, patch

import pytest

from verityrag.llm_providers import anthropic_complete_fn, gemini_complete_fn


def test_raises_clear_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        anthropic_complete_fn("What is the capital of France?")


def test_calls_sdk_with_expected_arguments_and_returns_text(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-test-key")

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Paris is the capital of France.")]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("anthropic.Anthropic", return_value=mock_client) as mock_anthropic_cls:
        result = anthropic_complete_fn("What is the capital of France?", model="claude-sonnet-5", max_tokens=256)

    mock_anthropic_cls.assert_called_once_with()  # no key passed explicitly -- SDK reads env var itself
    mock_client.messages.create.assert_called_once_with(
        model="claude-sonnet-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    assert result == "Paris is the capital of France."


def test_default_model_is_claude_sonnet_5(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-test-key")
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="answer")]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        anthropic_complete_fn("a prompt")  # no model= passed -- should use the default

    called_kwargs = mock_client.messages.create.call_args.kwargs
    assert called_kwargs["model"] == "claude-sonnet-5"


def test_integrates_with_llm_generator_end_to_end(monkeypatch):
    """The actual integration point: LLMGenerator (generation.py) calling
    this as its complete_fn, exactly as a real pipeline would use it."""
    from verityrag.generation import LLMGenerator

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-test-key")
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Nikola Tesla was born in Smiljan.")]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    class FakeEvidence:
        def __init__(self, chunk_id, text):
            self.chunk_id = chunk_id
            self.text = text

    with patch("anthropic.Anthropic", return_value=mock_client):
        generator = LLMGenerator(complete_fn=anthropic_complete_fn)
        result = generator.generate(
            "Where was Tesla born?",
            [FakeEvidence("c1", "Nikola Tesla was born in Smiljan in 1856.")],
            embedder=None,  # LLMGenerator doesn't use the embedder directly
        )

    assert result.text == "Nikola Tesla was born in Smiljan."
    assert result.generator == "llm"
    assert result.used_chunk_ids == ["c1"]
    # Confirm the evidence actually made it into the prompt sent to the model
    sent_prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Smiljan" in sent_prompt
    assert "Where was Tesla born?" in sent_prompt


# ---- gemini_complete_fn ----

def _make_client_error(code):
    """Builds a real google.genai.errors.ClientError with a given HTTP
    code, the same way the real SDK does, rather than a loosely-shaped
    mock -- so the test exercises the actual exception type
    gemini_complete_fn's except clause matches against."""
    from google.genai import errors as genai_errors
    return genai_errors.ClientError(code=code, response_json={"error": {"message": "mocked error"}})


def _make_server_error(code):
    """Same idea, but ServerError -- a REAL, distinct sibling class of
    ClientError (confirmed directly: neither is a subclass of the other,
    both are direct children of APIError). A real 503 from a live eval run
    showed that catching only ClientError meant a ServerError sailed
    straight through uncaught -- this helper exists so that gap has an
    actual regression test, not just a fixed except clause."""
    from google.genai import errors as genai_errors
    return genai_errors.ServerError(code=code, response_json={"error": {"message": "mocked server error"}})


def _make_daily_quota_error():
    """Reconstructs the EXACT structure of a real 429 hit in practice --
    same quotaId/quotaValue shape as the actual API response, not a
    simplified stand-in -- so the test verifies gemini_complete_fn's
    string-matching against the real field name, not an approximation of
    it. quotaValue=20 was the real daily cap observed for a brand-new
    model on a real account -- a genuinely restrictive, not hypothetical,
    number."""
    from google.genai import errors as genai_errors
    response_json = {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota...",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{
                        "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                        "quotaValue": "20",
                    }],
                },
            ],
        },
    }
    return genai_errors.ClientError(code=429, response_json=response_json)


def test_gemini_raises_clear_error_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gemini_complete_fn("What is the capital of France?")


def test_gemini_calls_sdk_and_returns_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")

    mock_response = MagicMock()
    mock_response.text = "Paris is the capital of France."
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        result = gemini_complete_fn("What is the capital of France?", model="gemini-2.5-flash")

    mock_client.models.generate_content.assert_called_once_with(
        model="gemini-2.5-flash", contents="What is the capital of France?"
    )
    assert result == "Paris is the capital of France."


def test_gemini_accepts_google_api_key_as_fallback(monkeypatch):
    """GOOGLE_API_KEY is accepted as an alternative to GEMINI_API_KEY --
    the SDK itself prefers GOOGLE_API_KEY when both are set (see
    llm_providers.py docstring); this only tests that gemini_complete_fn's
    OWN key-presence check doesn't reject GOOGLE_API_KEY-only setups."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-test-key")

    mock_response = MagicMock()
    mock_response.text = "answer"
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        result = gemini_complete_fn("a prompt")

    assert result == "answer"


def test_gemini_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setattr("time.sleep", lambda seconds: None)  # skip real waiting in the test

    mock_response = MagicMock()
    mock_response.text = "answer after retry"
    mock_client = MagicMock()
    # Fails with 429 twice, then succeeds on the third attempt
    mock_client.models.generate_content.side_effect = [
        _make_client_error(429),
        _make_client_error(429),
        mock_response,
    ]

    with patch("google.genai.Client", return_value=mock_client):
        result = gemini_complete_fn("a prompt", max_retries=5)

    assert result == "answer after retry"
    assert mock_client.models.generate_content.call_count == 3


def test_gemini_daily_quota_fails_fast_without_wasting_retries(monkeypatch):
    """Regression test for a real failure hit in practice: a full eval run
    hit a genuine 429 whose quotaId was
    'GenerateRequestsPerDayPerProjectPerModel-FreeTier' (quotaValue 20) --
    a DAILY cap, not a per-minute rate limit. The original code treated
    ANY 429 as retryable and burned all 5 attempts (~62s of backoff)
    before finally failing anyway, which is pure wasted time against a
    limit that can't clear within one script run (resets at midnight
    Pacific, not in seconds). This must fail on the FIRST attempt, not
    after exhausting the retry budget, and the resulting error must name
    the actual constraint (daily quota) rather than looking identical to
    an ordinary rate-limit failure."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _make_daily_quota_error()

    with patch("google.genai.Client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="[Dd]aily"):
            gemini_complete_fn("a prompt", max_retries=5)

    assert mock_client.models.generate_content.call_count == 1  # no retries wasted on an unclearable limit


def test_gemini_ordinary_429_without_daily_quota_text_still_retries(monkeypatch):
    """Confirms the daily-quota detection is specific to that case and
    doesn't accidentally swallow the ordinary, genuinely-retryable 429
    path -- a plain 429 with no 'PerDay' in its text (e.g. a per-minute
    rate limit) must still go through the normal retry-with-backoff flow,
    not the fail-fast path."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    mock_response = MagicMock()
    mock_response.text = "answer after ordinary rate limit cleared"
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        _make_client_error(429),  # generic 429, no quota details -- no "PerDay" text
        mock_response,
    ]

    with patch("google.genai.Client", return_value=mock_client):
        result = gemini_complete_fn("a prompt", max_retries=5)

    assert result == "answer after ordinary rate limit cleared"
    assert mock_client.models.generate_content.call_count == 2


def test_gemini_retries_on_503_server_error_then_succeeds(monkeypatch):
    """Regression test for a real failure hit in practice: a live eval run
    got a genuine 503 ('high demand') and crashed on the very first
    query, because ServerError (5xx) was never being caught at all -- only
    ClientError was. This uses a real ServerError instance, not
    ClientError, specifically to make sure the fix actually catches the
    right exception class and doesn't just look right by inspection."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    mock_response = MagicMock()
    mock_response.text = "answer after server recovered"
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        _make_server_error(503),
        _make_server_error(503),
        mock_response,
    ]

    with patch("google.genai.Client", return_value=mock_client):
        result = gemini_complete_fn("a prompt", max_retries=5)

    assert result == "answer after server recovered"
    assert mock_client.models.generate_content.call_count == 3


def test_gemini_gives_up_after_max_retries_on_persistent_503(monkeypatch):
    """A 503 that never clears should still eventually give up, not retry
    forever -- same bounded-retry guarantee as the 429 case, checked
    against ServerError specifically since it's now handled by the same
    code path as ClientError's 429."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _make_server_error(503)  # always 503

    with patch("google.genai.Client", return_value=mock_client):
        with pytest.raises(Exception):
            gemini_complete_fn("a prompt", max_retries=2)

    assert mock_client.models.generate_content.call_count == 3  # initial attempt + 2 retries


def test_gemini_404_raises_clear_error_without_retrying(monkeypatch):
    """Regression test for a real failure hit in practice: a deprecated
    model name returns 404, not 429 -- this must NOT be treated as a
    transient rate limit worth retrying (retrying a 404 just wastes 5
    rounds of exponential backoff arriving at the exact same permanent
    failure), and the resulting error must surface the API's original
    message, since in practice that message named the correct replacement
    model directly."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _make_client_error(404)

    with patch("google.genai.Client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="no longer available|404"):
            gemini_complete_fn("a prompt", model="gemini-2.5-flash", max_retries=5)

    assert mock_client.models.generate_content.call_count == 1  # no retries wasted on a permanent failure


def test_gemini_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _make_client_error(429)  # always 429

    with patch("google.genai.Client", return_value=mock_client):
        with pytest.raises(Exception):  # google.genai.errors.ClientError, re-raised after exhausting retries
            gemini_complete_fn("a prompt", max_retries=2)

    assert mock_client.models.generate_content.call_count == 3  # initial attempt + 2 retries


def test_gemini_does_not_retry_on_non_rate_limit_errors():
    """A 400 (bad request) or similar should fail immediately, not burn
    through retries waiting for a rate limit to clear that was never the
    actual problem."""
    import os
    os.environ["GEMINI_API_KEY"] = "fake-test-key"
    try:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = _make_client_error(400)

        with patch("google.genai.Client", return_value=mock_client):
            with pytest.raises(Exception):
                gemini_complete_fn("a prompt", max_retries=5)

        assert mock_client.models.generate_content.call_count == 1  # no retries attempted
    finally:
        del os.environ["GEMINI_API_KEY"]


def test_gemini_integrates_with_llm_generator_end_to_end(monkeypatch):
    from verityrag.generation import LLMGenerator

    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    mock_response = MagicMock()
    mock_response.text = "Nikola Tesla was born in Smiljan."
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    class FakeEvidence:
        def __init__(self, chunk_id, text):
            self.chunk_id = chunk_id
            self.text = text

    with patch("google.genai.Client", return_value=mock_client):
        generator = LLMGenerator(complete_fn=gemini_complete_fn)
        result = generator.generate(
            "Where was Tesla born?",
            [FakeEvidence("c1", "Nikola Tesla was born in Smiljan in 1856.")],
            embedder=None,
        )

    assert result.text == "Nikola Tesla was born in Smiljan."
    assert result.generator == "llm"
    sent_prompt = mock_client.models.generate_content.call_args.kwargs["contents"]
    assert "Smiljan" in sent_prompt