"""
Planner role: decide whether a question is really several questions.

Rule-based and deterministic on purpose: no LLM call, no added latency, and
every decision is unit-testable offline. It splits only when EVERY part is a
complete question on its own:

  "When was Tesla born? Also, where did Edison die?"        -> 2 parts (sentences)
  "When was Tesla born, and where did Edison die?"           -> 2 parts (conjunction)
  "What is the capital of France and who is its president?"  -> left whole: "its"
                                                                points back at the
                                                                first part, so the
                                                                second can't stand alone
  "What are the pros and cons of solar power?"               -> left whole: one question

Anything it isn't sure about stays whole, so a single question is never
split and behaves exactly as it did before the planner existed. Dependent
sub-questions (pronoun or "what happened next" back-references) are the case
a rule can't handle; that is where an LLM planner would plug in.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_PARTS = 4

_PREP = r"(?:in|on|at|by|for|from|of|to|with|during|after|before|under|since|until|between|among|through|about)"
_WH = r"(?:what|which|who|whom|whose|when|where|why|how)"
_AUX = r"(?:is|are|was|were|do|does|did|can|could|will|would|has|have|had|should|may|might)"

_STARTS_QUESTION = re.compile(rf"^(?:{_PREP}\s+)?(?:{_WH}|{_AUX})\b", re.I)
_ANY_WH = re.compile(rf"\b{_WH}\b", re.I)
_STARTS_SUBJECT_WH = re.compile(r"^(?:who|what|when|where)\b", re.I)
_CONNECTOR = re.compile(r"^(?:also|and|then|plus|additionally|next)\b[\s,:;-]*", re.I)
# A conjunction (or bare semicolon) that is followed by a wh-question start.
_BOUNDARY = re.compile(
    rf"\s*[,;]?\s*\b(?:and|also|plus)\b\s+(?=(?:{_PREP}\s+)?{_WH}\b)|\s*;\s*(?=(?:{_PREP}\s+)?{_WH}\b)",
    re.I,
)
_PRONOUN = re.compile(r"\b(?:he|she|it|they|him|her|them|his|its|their)\b", re.I)
_ANAPHORA = re.compile(r"\b(?:next|then|else|same|former|latter|above|previous|later|afterwards|thereafter)\b", re.I)


@dataclass
class Plan:
    sub_questions: list[str]
    rule: str = "single"  # "single" | "sentences" | "conjunction"


def _has_own_entity(part: str) -> bool:
    """A capitalised word (not the first) or a number means the part names its
    own subject, so a pronoun in it can refer to something inside it."""
    return any(t[:1].isupper() or any(ch.isdigit() for ch in t) for t in part.split()[1:])


def _standalone(part: str, first: bool) -> bool:
    words = part.split()
    # "Sky Movies also include what optional soundtracks?" has its wh-word mid-sentence.
    if not (_STARTS_QUESTION.match(part) or _ANY_WH.search(part)):
        return False
    # Fragments like "in which city?" are too short to be a question alone.
    if len(words) < (3 if _STARTS_SUBJECT_WH.match(part) else 4):
        return False
    if first:
        return True
    if _ANAPHORA.search(part):
        return False
    return not (_PRONOUN.search(part) and not _has_own_entity(part))


def _clean(piece: str) -> str:
    return piece.strip().rstrip("?.,;").strip() + "?"


def _cap(part: str) -> str:
    return part[:1].upper() + part[1:]


class Planner:
    def plan(self, question: str) -> Plan:
        q = " ".join(question.split())
        whole = Plan([question], "single")

        # 1) Separate sentences: "A? Also, b?"
        frags = re.split(r"(?<=\?)\s+", q)
        if len(frags) > 1:
            parts = [frags[0]] + [_CONNECTOR.sub("", f, count=1).strip() for f in frags[1:]]
            parts = [_clean(p) for p in parts]
            if len(parts) <= MAX_PARTS and all(_standalone(p, i == 0) for i, p in enumerate(parts)):
                return Plan([_cap(p) for p in parts], "sentences")
            return whole

        # 2) One sentence joined by a conjunction: "A, and b?"
        pieces, start = [], 0
        for m in _BOUNDARY.finditer(q):
            pieces.append(q[start:m.start()])
            start = m.end()
        pieces.append(q[start:])
        if len(pieces) < 2 or len(pieces) > MAX_PARTS:
            return whole
        parts = [_clean(p) for p in pieces]
        if all(_standalone(p, i == 0) for i, p in enumerate(parts)):
            return Plan([_cap(p) for p in parts], "conjunction")
        return whole
