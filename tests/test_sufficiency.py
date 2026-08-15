from verityrag.sufficiency import (
    CalibrationExample,
    SufficiencyGate,
    extract_features,
    train_sufficiency_gate,
)


class FakeCandidate:
    def __init__(self, dense_score, lexical_score=1.0):
        self.dense_score = dense_score
        self.lexical_score = lexical_score


def test_default_gate_thresholds_on_top1_dense_score():
    gate = SufficiencyGate(threshold=0.2, lexical_threshold=1.0)
    assert gate.is_sufficient([FakeCandidate(0.3, lexical_score=0.0)]) is True
    assert gate.is_sufficient([FakeCandidate(0.1, lexical_score=0.0)]) is False


def test_default_gate_or_combines_dense_and_lexical_signal():
    gate = SufficiencyGate(threshold=0.2, lexical_threshold=1.0)
    # weak dense score but strong, unambiguous lexical match -> still sufficient
    assert gate.is_sufficient([FakeCandidate(0.05, lexical_score=1.5)]) is True
    # both weak -> insufficient
    assert gate.is_sufficient([FakeCandidate(0.05, lexical_score=0.1)]) is False


def test_default_gate_handles_empty_candidates():
    gate = SufficiencyGate(threshold=0.2)
    assert gate.is_sufficient([]) is False


def test_extract_features_shape_and_empty_case():
    feats_empty = extract_features([])
    assert feats_empty == [0.0, 0.0, 0.0, 0.0, 0.0]
    feats = extract_features([FakeCandidate(0.3), FakeCandidate(0.2), FakeCandidate(0.1)])
    assert len(feats) == 5
    assert feats[0] == 0.3  # top1_dense


def _synthetic_calibration_set(n=40):
    """Synthetic but clearly-separable data purely to test the training
    mechanics (fit/predict pipeline), not to validate real-world accuracy --
    that's what eval.py's calibration on real SQuAD-derived questions is for."""
    examples = []
    for i in range(n):
        pos_feats = extract_features([FakeCandidate(0.5, 3.0), FakeCandidate(0.45, 2.5)])
        neg_feats = extract_features([FakeCandidate(0.05, 0.1), FakeCandidate(0.02, 0.05)])
        examples.append(CalibrationExample(features=pos_feats, label=1))
        examples.append(CalibrationExample(features=neg_feats, label=0))
    return examples


def test_train_sufficiency_gate_separates_synthetic_classes():
    examples = _synthetic_calibration_set()
    gate = train_sufficiency_gate(examples)
    assert gate.is_calibrated is True
    assert gate.is_sufficient([FakeCandidate(0.5, 3.0), FakeCandidate(0.45, 2.5)]) is True
    assert gate.is_sufficient([FakeCandidate(0.05, 0.1), FakeCandidate(0.02, 0.05)]) is False


def test_train_sufficiency_gate_rejects_too_few_examples():
    import pytest
    with pytest.raises(ValueError):
        train_sufficiency_gate([CalibrationExample(features=[0.1] * 5, label=1)] * 3)


def test_train_sufficiency_gate_rejects_single_class():
    import pytest
    examples = [CalibrationExample(features=[0.1] * 5, label=1) for _ in range(10)]
    with pytest.raises(ValueError):
        train_sufficiency_gate(examples)
