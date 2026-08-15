"""
Grounding / hallucination detection.

Approach: claim decomposition + evidence matching -- the same conceptual
approach used by real faithfulness metrics (e.g. RAGAS "faithfulness",
FActScore): break a generated answer into atomic claims (sentences), then
score how well each claim is supported by the retrieved evidence set, using
two complementary signals:

  1. semantic similarity  -- max cosine similarity between the claim and any
     single evidence chunk (via the same embedder used for retrieval)
  2. lexical grounding    -- fraction of the claim's informative (non-stopword)
     tokens that literally appear in the evidence, which catches numeric/
     entity fabrication that a purely semantic score can miss (e.g. a claim
     that is topically similar to the evidence but states the wrong number
     or wrong name)

A claim is only trusted if BOTH signals clear their thresholds -- requiring
agreement between a semantic and a lexical view is what lets this catch
"plausible-sounding but wrong" fabrications, not just off-topic ones.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .chunking import split_sentences
from .embedding import Embedder, cosine_sim_matrix

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to", "for",
    "and", "or", "as", "by", "with", "at", "from", "that", "this", "it", "its",
    "be", "has", "have", "had", "which", "who", "whom", "their", "his", "her",
}


def _informative_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


@dataclass
class ClaimGrounding:
    claim: str
    semantic_support: float
    lexical_support: float
    is_grounded: bool
    best_evidence_chunk_id: str | None


@dataclass
class GroundingReport:
    claims: list[ClaimGrounding]
    overall_score: float  # fraction of claims grounded
    ungrounded_claims: list[str]
    verdict: str  # "grounded" | "partially_grounded" | "ungrounded"


def check_grounding(
    answer: str,
    evidence: list,  # list of RetrievedChunk-like objects with .text and .chunk_id
    embedder: Embedder,
    semantic_threshold: float = 0.15,
    lexical_threshold: float = 0.4,
) -> GroundingReport:
    claims = split_sentences(answer)
    if not claims:
        return GroundingReport(claims=[], overall_score=0.0, ungrounded_claims=[], verdict="ungrounded")
    if not evidence:
        report_claims = [
            ClaimGrounding(claim=c, semantic_support=0.0, lexical_support=0.0,
                            is_grounded=False, best_evidence_chunk_id=None)
            for c in claims
        ]
        return GroundingReport(claims=report_claims, overall_score=0.0,
                                ungrounded_claims=claims, verdict="ungrounded")

    evidence_texts = [e.text for e in evidence]
    evidence_ids = [e.chunk_id for e in evidence]
    evidence_token_sets = [_informative_tokens(t) for t in evidence_texts]

    claim_vecs = embedder.embed(claims)
    evidence_vecs = embedder.embed(evidence_texts)
    sim_matrix = cosine_sim_matrix(claim_vecs, evidence_vecs)  # [n_claims, n_evidence]

    results: list[ClaimGrounding] = []
    for i, claim in enumerate(claims):
        sem_scores = sim_matrix[i]
        best_idx = int(np.argmax(sem_scores)) if sem_scores.size else -1
        semantic_support = float(sem_scores[best_idx]) if best_idx >= 0 else 0.0

        claim_tokens = _informative_tokens(claim)
        if claim_tokens and best_idx >= 0:
            lexical_support = len(claim_tokens & evidence_token_sets[best_idx]) / len(claim_tokens)
        else:
            lexical_support = 0.0

        is_grounded = semantic_support >= semantic_threshold and lexical_support >= lexical_threshold
        results.append(ClaimGrounding(
            claim=claim,
            semantic_support=round(semantic_support, 4),
            lexical_support=round(lexical_support, 4),
            is_grounded=is_grounded,
            best_evidence_chunk_id=evidence_ids[best_idx] if best_idx >= 0 else None,
        ))

    grounded_count = sum(1 for r in results if r.is_grounded)
    overall = grounded_count / len(results)
    ungrounded = [r.claim for r in results if not r.is_grounded]

    if overall >= 0.99:
        verdict = "grounded"
    elif overall <= 0.0:
        verdict = "ungrounded"
    else:
        verdict = "partially_grounded"

    return GroundingReport(claims=results, overall_score=round(overall, 4),
                            ungrounded_claims=ungrounded, verdict=verdict)
