"""
Sufficiency classifier: decides "do we have enough relevant evidence to even
attempt an answer" -- the agent's FIRST line of defense against hallucination
(the grounding checker in grounding.py is the SECOND line, guarding the
generator's output; this module guards entry into generation at all).

Why not a single hand-picked threshold on top-1 cosine score: empirically
(see eval.py / README "Calibration"), the raw top-1 dense score for genuinely
on-topic queries and for off-topic-but-coincidentally-keyword-overlapping
queries overlap substantially when using a hashed bag-of-n-grams embedding
(no pretrained neural encoder available in this build environment -- see
embedding.py). No single threshold on that one feature cleanly separates the
two classes. Multiple retrieval-time features considered together separate
them better, which is exactly the shape of a small binary classification
problem -- so that's what this is, trained on labeled
answerable/should-abstain examples via `calibrate()`.

Falls back to a plain single-threshold rule when no calibration data is
available (e.g. a fresh index with nothing to calibrate against yet) --
calibration is an optional upgrade, not a hard requirement to run the system.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from .retrieval import RetrievedChunk

FEATURE_NAMES = [
    "top1_dense",
    "top3_mean_dense",
    "top1_lexical_raw",
    "dense_gap_top1_top2",
    "n_candidates_above_floor",
]


def extract_features(candidates: list[RetrievedChunk], floor: float = 0.08) -> list[float]:
    if not candidates:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    dense = [c.dense_score for c in candidates]
    top1_dense = dense[0]
    top3_mean_dense = float(np.mean(dense[:3]))
    top1_lexical_raw = candidates[0].lexical_score
    gap = dense[0] - dense[1] if len(dense) > 1 else dense[0]
    n_above_floor = sum(1 for d in dense if d >= floor)
    return [top1_dense, top3_mean_dense, top1_lexical_raw, gap, float(n_above_floor)]


@dataclass
class CalibrationExample:
    features: list[float]
    label: int  # 1 = should attempt to answer, 0 = should abstain


class SufficiencyGate:
    """
    Default (uncalibrated) gate. Sufficient if EITHER the raw dense cosine
    score OR the raw BM25 lexical score clears its own threshold -- an OR,
    not just dense alone. Dense (hashed bag-of-n-grams, no IDF weighting)
    can under-score a short, unambiguous, single-clear-match document simply
    because there are few hashed n-grams to overlap on; BM25 already applies
    IDF down-weighting internally, so a real BM25 signal above the noise
    floor is meaningful evidence on its own even when the dense signal is
    weak. Calibrate a CalibratedSufficiencyGate (below) when labeled data is
    available -- it learns the right combination instead of two hand-picked
    thresholds.
    """

    def __init__(self, threshold: float = 0.15, lexical_threshold: float = 1.0):
        self.threshold = threshold
        self.lexical_threshold = lexical_threshold
        self.is_calibrated = False

    def is_sufficient(self, candidates: list[RetrievedChunk]) -> bool:
        if not candidates:
            return False
        top = candidates[0]
        return top.dense_score >= self.threshold or top.lexical_score >= self.lexical_threshold

    def score(self, candidates: list[RetrievedChunk]) -> float:
        """Coarse [0,1] confidence proxy for the uncalibrated fallback gate --
        used only to decide whether a reformulation attempt is worth it.
        Scaled so it reaches 1.0 exactly when is_sufficient() would be True."""
        if not candidates:
            return 0.0
        top = candidates[0]
        return max(top.dense_score / max(self.threshold, 1e-9),
                    top.lexical_score / max(self.lexical_threshold, 1e-9))


class CalibratedSufficiencyGate(SufficiencyGate):
    """Multi-feature classifier, trained on labeled answerable/should-abstain examples."""

    def __init__(self, model: lgb.LGBMClassifier, decision_threshold: float = 0.5):
        super().__init__(threshold=float("nan"))
        self.model = model
        self.decision_threshold = decision_threshold
        self.is_calibrated = True

    def is_sufficient(self, candidates: list[RetrievedChunk]) -> bool:
        features = np.array([extract_features(candidates)])
        prob = self.model.predict_proba(features)[0, 1]
        return bool(prob >= self.decision_threshold)

    def score(self, candidates: list[RetrievedChunk]) -> float:
        features = np.array([extract_features(candidates)])
        return float(self.model.predict_proba(features)[0, 1])


def train_sufficiency_gate(examples: list[CalibrationExample]) -> CalibratedSufficiencyGate:
    if len(examples) < 8:
        raise ValueError("Need at least 8 calibration examples (both classes represented)")
    X = np.array([e.features for e in examples])
    y = np.array([e.label for e in examples])
    if len(set(y.tolist())) < 2:
        raise ValueError("Calibration examples must include both classes (answerable and should-abstain)")

    model = lgb.LGBMClassifier(
        n_estimators=40,
        num_leaves=7,
        learning_rate=0.1,
        min_child_samples=2,
        verbosity=-1,
    )
    model.fit(X, y)
    return CalibratedSufficiencyGate(model)
