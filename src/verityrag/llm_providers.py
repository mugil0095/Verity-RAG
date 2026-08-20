"""
Concrete complete_fn implementations for LLMGenerator (generation.py) --
this is the "swap in a real model" step. LLMGenerator's interface is
already provider-agnostic (any Callable[[str], str]); this module supplies
concrete, ready-to-use implementations rather than leaving that as an
exercise. Two are provided:

  anthropic_complete_fn -- calls the real Anthropic API. No free tier
    (a small starter credit only) -- see README "Swapping in a real LLM".

  gemini_complete_fn -- calls Google's Gemini API, which DOES have a real,
    ongoing, no-credit-card-required free tier (confirmed against current
    docs/reporting as of this writing, not assumed -- Google has changed
    these limits multiple times through 2026, so if you hit persistent
    429s, check https://ai.google.dev/gemini-api/docs/rate-limits for the
    current numbers rather than trusting a hardcoded figure here). This is
    the one to reach for if you don't want to spend anything to try real
    LLM generation. Model IDs in this API also go stale fast -- observed
    directly, not hypothetically: the original default here
    (gemini-2.5-flash) returned a 404 "no longer available to new users"
    within days. gemini_complete_fn surfaces the API's own error message on
    a 404, which in practice named the correct replacement directly.

Why this matters for the project's actual thesis: ExtractiveGenerator is
grounded by construction (it can only copy evidence text verbatim), which
is a safe default but means the grounding checker (grounding.py) has never
actually been tested against a REAL hallucination -- only synthetic ones in
unit tests. A real LLM can say something plausible-sounding that isn't in
the evidence even when the evidence is right there in its context window.
Wiring this in is what makes the hallucination-detection story real instead
of structurally guaranteed to pass.

Both are optional dependencies -- not force-imported at module load,
matching how sentence-transformers is handled in embedding.py: only
imported when the specific function you're using is actually called, so
neither adds a hard dependency for users who need only the other (or
neither). Both read their API key from an environment variable -- NEVER
hardcode a key in source or pass one as a literal string anywhere.
"""
from __future__ import annotations

import os
import time


def anthropic_complete_fn(prompt: str, model: str = "claude-sonnet-5", max_tokens: int = 512) -> str:
    """A complete_fn for LLMGenerator that calls the real Anthropic API.

    Usage:
        from verityrag.generation import LLMGenerator
        from verityrag.llm_providers import anthropic_complete_fn
        pipeline = VerityRAGPipeline(generator=LLMGenerator(complete_fn=anthropic_complete_fn))

    Requires `pip install anthropic` and an ANTHROPIC_API_KEY environment
    variable set (get a key at https://console.anthropic.com/). Raises a
    clear, actionable error immediately if either is missing, rather than
    failing confusingly deep inside a query with a generic exception.

    No ongoing free tier -- new accounts get a small starter credit, after
    that it's pay-per-token. If that's not workable for you right now, see
    gemini_complete_fn below instead, which has a genuine free tier.

    model defaults to claude-sonnet-5 -- Anthropic's "best balance of
    intelligence and speed for most production workloads" tier as of this
    writing (verified against current docs, not assumed from training
    data, since model names change). Override if you want a different
    tier; see https://docs.claude.com/en/docs/about-claude/models/overview
    for the current lineup.
    """
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "anthropic_complete_fn requires the anthropic package, which "
            "isn't installed. Install it with: pip install anthropic"
        ) from e

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. Get a key "
            "at https://console.anthropic.com/ and set it before using "
            "anthropic_complete_fn -- e.g. on macOS/Linux: "
            "`export ANTHROPIC_API_KEY=sk-ant-...`; on Windows PowerShell: "
            '`$env:ANTHROPIC_API_KEY="sk-ant-..."`. Never hardcode the key '
            "in source or commit it -- environment variable only."
        )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def gemini_complete_fn(prompt: str, model: str = "gemini-3.6-flash", max_retries: int = 5) -> str:
    """A complete_fn for LLMGenerator that calls the real Google Gemini API
    -- the free option. No credit card required to get started (verified
    against current docs, not assumed).

    Usage:
        from verityrag.generation import LLMGenerator
        from verityrag.llm_providers import gemini_complete_fn
        pipeline = VerityRAGPipeline(generator=LLMGenerator(complete_fn=gemini_complete_fn))

    Requires `pip install google-genai` and a GEMINI_API_KEY environment
    variable set (get a free key at https://aistudio.google.com/apikey --
    no card needed). Raises a clear, actionable error immediately if either
    is missing.

    Free-tier rate limits are real and modest (roughly 10-15 requests/min
    depending on model and date -- Google has revised these multiple times
    through 2026, check https://ai.google.dev/gemini-api/docs/rate-limits
    for the current numbers rather than trusting any specific figure here).
    A full eval.py run makes ~150 sequential calls, which WILL likely hit
    the rate limit partway through -- this function retries with
    exponential backoff on a 429 AND on any 5xx server error (up to
    max_retries times) specifically so a full run doesn't just crash the
    first time that happens. The 5xx case is not hypothetical -- a live
    eval run hit a genuine 503 ("high demand, try again later") on its
    very first query, and crashed immediately, because the retry logic at
    the time only caught ClientError (4xx) and ServerError (5xx) are
    sibling classes in this SDK, neither a subclass of the other. Fixed by
    catching the shared APIError base class instead. If you hit
    max_retries, the run genuinely needs to wait longer between requests
    than backoff alone provides -- that's real free-tier friction, not a
    bug in this function.

    model defaults to gemini-3.6-flash, current as of this writing and
    confirmed still free-tier-eligible -- but model names in this API
    genuinely do go stale fast: the previous default (gemini-2.5-flash)
    stopped being available to new accounts within the same week this
    function was written, discovered via a real 404 from the live API, not
    predicted in advance. If you hit a 404 "model no longer available"
    error, see the note on that below -- the API's own error message
    usually names the current replacement directly.
    """
    try:
        from google import genai
        from google.genai import errors as genai_errors
    except ImportError as e:
        raise ImportError(
            "gemini_complete_fn requires the google-genai package, which "
            "isn't installed. Install it with: pip install google-genai"
        ) from e

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. Get a FREE "
            "key (no credit card required) at https://aistudio.google.com/apikey "
            "and set it before using gemini_complete_fn -- e.g. on "
            "macOS/Linux: `export GEMINI_API_KEY=...`; on Windows "
            'PowerShell: `$env:GEMINI_API_KEY="..."`. Never hardcode the '
            "key in source or commit it -- environment variable only."
        )

    client = genai.Client()  # auto-reads GEMINI_API_KEY/GOOGLE_API_KEY from the environment

    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text
        except genai_errors.APIError as e:
            # Catch the SHARED base class, not ClientError specifically --
            # a real 503 ("high demand") from a live run showed this gap
            # directly: ServerError (5xx) and ClientError (4xx) are SIBLING
            # classes in this SDK, neither a subclass of the other, so
            # catching only ClientError meant a 503 sailed straight through
            # uncaught and crashed an entire eval run on the very first
            # query. Catching APIError here covers both and any other
            # subclass this SDK adds later, so this category of gap
            # shouldn't repeat for a different error type next time.
            code = getattr(e, "code", None)
            if code == 404:
                # This API deprecates model IDs often enough that this is a
                # real, expected failure mode, not an edge case -- observed
                # directly: gemini-2.5-flash returned exactly this 404 within
                # days of being set as the default here. Google's error text
                # usually names the current replacement model directly (it
                # did in that case), so surface the FULL original message
                # rather than a bare traceback or a generic wrapper that
                # would hide the one detail that actually tells you what to
                # do next.
                raise RuntimeError(
                    f"Model '{model}' was rejected by the Gemini API (404) -- "
                    f"likely deprecated for new accounts. The API's error "
                    f"message below often names the current replacement "
                    f"directly; pass that as model= next time, or check "
                    f"https://ai.google.dev/gemini-api/docs/models for the "
                    f"current lineup.\n\nOriginal error: {e}"
                ) from e
            if code == 429 and "PerDay" in str(e):
                # A DAILY quota, not a per-minute rate limit -- confirmed
                # directly from a real 429 in practice: quotaId
                # 'GenerateRequestsPerDayPerProjectPerModel-FreeTier',
                # quotaValue 20. This distinction matters enormously:
                # retrying with backoff (seconds) inside one script run
                # cannot fix a limit that only resets at midnight Pacific
                # time. The original code treated this as a retryable 429
                # and burned all 5 attempts (~62s of backoff) before
                # finally failing anyway -- pure wasted time against a
                # limit no amount of waiting-within-this-run clears. Fail
                # immediately instead, with the actual constraint named.
                raise RuntimeError(
                    f"Daily free-tier quota exhausted for model '{model}' -- "
                    f"this is a PER-DAY limit (resets at midnight Pacific "
                    f"time), not a per-minute one, so retrying within this "
                    f"run cannot help. Third-party sources report wildly "
                    f"different daily caps for different Gemini models (from "
                    f"~20 to 1,500+), and a brand-new model like this one may "
                    f"start with a much tighter quota than an established one "
                    f"-- check your actual current limit at "
                    f"https://aistudio.google.com/ rather than trusting any "
                    f"number found elsewhere, since it varies by account, "
                    f"region, and model. A '-lite' variant, if one exists for "
                    f"the current model generation, has historically had a "
                    f"materially higher free daily quota than the full model "
                    f"-- worth trying, but verify it against your own account "
                    f"rather than assuming.\n\nOriginal error: {e}"
                ) from e
            # 429 (rate limit) and any 5xx (transient server-side issue,
            # e.g. the 503 "high demand" seen in practice) are both worth
            # retrying with backoff. Any OTHER 4xx (400 bad request, 401/403
            # auth failure, etc.) is typically permanent -- retrying one
            # just burns the whole backoff budget arriving at the same
            # failure, so those still fail immediately, same as 404.
            is_retryable = code == 429 or (isinstance(code, int) and 500 <= code < 600)
            if not is_retryable or attempt == max_retries:
                raise
            time.sleep(delay)
            delay *= 2  # exponential backoff: 2s, 4s, 8s, 16s, 32s

    raise RuntimeError("unreachable")  # loop always returns or raises above