"""
Logs real user input from the deployed demo (questions asked, documents
added) into a separate, private GitHub repo -- not this project's own
repo. One file per submission (timestamp + random suffix), so concurrent
visitors never race over the same file the way updating one shared,
growing log file would.

Fires the actual GitHub commit in a background thread -- log_input()
itself returns immediately. This app's whole thesis is real-time
response; a synchronous requests.put() with a 10s timeout would mean a
slow GitHub API response (or a network hiccup) adds up to 10 seconds of
delay to every single question or document submission, for a tracking
call that has nothing to do with the actual answer. That's a real,
user-facing latency cost for an optional side feature, not acceptable.

Deliberately fails silently, never raises into the caller (and now,
never blocks it either): this is an optional tracking feature, not core
to the app working. A visitor asking a question should never see an
error, or feel a delay, because a logging call to an unrelated service
failed or was slow for any reason (missing token, network issue, GitHub
rate limit, etc.). Failures are logged via Python's own logging module
(visible in Streamlit Community Cloud's app logs to whoever has repo
write access) rather than surfaced to the visitor.

Requires a GITHUB_TOKEN in Streamlit secrets (Settings -> Secrets on
Streamlit Community Cloud, or a local .streamlit/secrets.toml for
testing) with write access to the target repo -- a personal access
token needs the `repo` scope (classic) or "Contents" repository
permission: write (fine-grained). Never hardcode this token in source;
never paste it into a chat or commit it anywhere. If the secret isn't
configured, logging is silently skipped rather than treated as an error
-- this makes local development and any environment without the secret
configured work normally, without needing every contributor to have
write access to a private tracking repo just to run the app.
"""
from __future__ import annotations

import base64
import json
import logging
import secrets
import threading
import time

import requests

logger = logging.getLogger(__name__)

TRACKER_REPO = "tracker-data"  # separate, private repo -- NOT this project's own repo
TRACKER_OWNER = "mugil0095"  # public GitHub username, not sensitive -- unlike the token, doesn't need a secret
TRACKER_FOLDER = "verityrag"  # subfolder within that repo, alongside mu-tracker's own data
GITHUB_API_BASE = "https://api.github.com"


def _get_secret(key: str) -> str | None:
    """Reads from Streamlit secrets if available; returns None (not an
    exception) if secrets aren't configured at all -- e.g. local dev
    without a secrets.toml file, or any environment where this optional
    feature simply isn't set up yet. Must be called from the main
    Streamlit script-run thread, not from the background thread
    log_input() spawns -- see log_input()'s own docstring for why."""
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None


def _commit_to_github(token: str, input_type: str, data: dict) -> bool:
    """The actual, potentially-slow network call -- deliberately takes the
    token as a plain argument rather than reading it itself, so this
    function has no Streamlit dependency at all and is safe to run on any
    thread, including the background one log_input() spawns.

    Returns True if the commit succeeded, False otherwise (never raises)
    -- exists as its own testable unit; log_input() itself doesn't wait
    for or use this return value, since it runs in the background."""
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
    unique_suffix = secrets.token_hex(4)  # avoids any collision even for near-simultaneous submissions
    filename = f"{TRACKER_FOLDER}/{input_type}_{timestamp}_{unique_suffix}.json"

    payload = {"logged_at_utc": timestamp, "type": input_type, **data}
    content_b64 = base64.b64encode(json.dumps(payload, indent=2).encode("utf-8")).decode("ascii")

    url = f"{GITHUB_API_BASE}/repos/{TRACKER_OWNER}/{TRACKER_REPO}/contents/{filename}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    body = {
        "message": f"Log {input_type} from VerityRAG demo",
        "content": content_b64,
    }

    try:
        response = requests.put(url, headers=headers, json=body, timeout=10)
        if response.status_code not in (200, 201):
            logger.warning(
                "Usage tracking failed (status %s): %s",
                response.status_code, response.text[:300],
            )
            return False
        return True
    except requests.RequestException as e:
        logger.warning("Usage tracking failed (network error): %s", e)
        return False


def log_input(input_type: str, data: dict) -> None:
    """input_type: a short label, e.g. "question" or "document" -- becomes
    part of the stored filename for easy browsing in the tracker repo.
    data: whatever's relevant to log for this input type -- gets stored
    as-is (plus a timestamp) in the committed JSON file.

    Returns immediately, always -- the actual GitHub commit happens on a
    background thread. Reads the secret HERE, on the calling (main)
    thread, before spawning that thread: st.secrets needs Streamlit's
    script-run context, which a background thread doesn't have, so the
    background thread only ever does plain, Streamlit-free HTTP work via
    _commit_to_github(). Callers should treat this as fire-and-forget --
    there's no return value to check, and none of this should ever be
    treated as something to show the visitor or block on."""
    token = _get_secret("GITHUB_TOKEN")
    if not token:
        logger.debug("Usage tracking skipped: GITHUB_TOKEN not configured in Streamlit secrets.")
        return

    thread = threading.Thread(target=_commit_to_github, args=(token, input_type, data), daemon=True)
    thread.start()