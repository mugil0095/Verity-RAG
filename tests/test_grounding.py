from verityrag.embedding import HashingEmbedder
from verityrag.grounding import check_grounding


class FakeEvidence:
    """Minimal stand-in for RetrievedChunk -- grounding only needs .text/.chunk_id."""
    def __init__(self, chunk_id, text):
        self.chunk_id = chunk_id
        self.text = text


EVIDENCE = [
    FakeEvidence("c1", "Nikola Tesla was born on 10 July 1856 in Smiljan, in the Austrian Empire."),
    FakeEvidence("c2", "Tesla is best known for his contributions to the design of the modern alternating current electricity supply system."),
]

EMBEDDER = HashingEmbedder(n_features=4096)


def test_fully_supported_answer_scores_high():
    answer = "Tesla was born on 10 July 1856 in Smiljan. He contributed to the alternating current system."
    report = check_grounding(answer, EVIDENCE, EMBEDDER)
    assert report.overall_score >= 0.99
    assert report.verdict == "grounded"
    assert report.ungrounded_claims == []


def test_fabricated_claim_scores_low():
    # A claim with an entity/fact not present anywhere in the evidence at all.
    answer = "Tesla won the Nobel Prize in Chemistry in 1975 for his work on synthetic rubber."
    report = check_grounding(answer, EVIDENCE, EMBEDDER)
    assert report.overall_score < 0.5
    assert len(report.ungrounded_claims) >= 1


def test_partially_supported_answer_flags_specific_span():
    answer = (
        "Tesla was born on 10 July 1856 in Smiljan. "
        "He later became the mayor of New York City in 1920."
    )
    report = check_grounding(answer, EVIDENCE, EMBEDDER)
    assert report.verdict == "partially_grounded"
    grounded_flags = [c.is_grounded for c in report.claims]
    assert True in grounded_flags and False in grounded_flags


def test_no_evidence_means_ungrounded():
    answer = "Tesla was born in 1856."
    report = check_grounding(answer, [], EMBEDDER)
    assert report.overall_score == 0.0
    assert report.verdict == "ungrounded"


def test_empty_answer_handled_gracefully():
    report = check_grounding("", EVIDENCE, EMBEDDER)
    assert report.claims == []
    assert report.verdict == "ungrounded"
