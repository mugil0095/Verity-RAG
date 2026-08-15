"""
Query reformulation for the agent's retry hop.

Uses pseudo-relevance feedback (PRF): a classic IR technique where, if the
first retrieval pass came back weak, you expand the query with informative
terms drawn from the (best-available, if imperfect) top candidate, then
retry. This is a real, well-established technique -- not a placeholder --
and it needs no external model.
"""
from __future__ import annotations

import re

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to", "for",
    "and", "or", "as", "by", "with", "at", "from", "what", "when", "where",
    "who", "how", "why", "did", "does", "do", "which",
}


def _keywords(text: str, limit: int) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    seen = []
    for t in tokens:
        if t in _STOPWORDS or len(t) < 3:
            continue
        if t not in seen:
            seen.append(t)
        if len(seen) >= limit:
            break
    return seen


def reformulate_query(original_query: str, current_query: str, weak_candidates: list) -> str:
    """
    If we have at least one (weak) candidate, pull expansion terms from it;
    otherwise fall back to re-emphasizing the original query's own keywords
    (handles the "nothing at all matched" case, e.g. a genuinely out-of-domain
    question, where PRF has no signal to draw from).
    """
    base_terms = _keywords(original_query, limit=6)

    if weak_candidates:
        top_text = weak_candidates[0].text
        expansion_terms = [t for t in _keywords(top_text, limit=5) if t not in base_terms]
    else:
        expansion_terms = []

    expanded = " ".join(base_terms + expansion_terms[:4])
    return expanded if expanded.strip() else original_query
