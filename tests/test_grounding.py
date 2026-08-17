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

def test_claim_grounded_despite_dilution_from_unrelated_sentences_in_same_chunk():
    """Regression test for a real, measured effect: a claim matching one
    sentence inside a multi-sentence chunk scored only 0.41 similarity
    against the WHOLE chunk (diluted by 4 unrelated sentences) vs 1.0
    against that sentence in isolation -- a 2.4x dilution, measured
    directly with HashingEmbedder before this fix. Sentence-level matching
    (not whole-chunk) is what avoids it -- this is why avg_grounding_score
    measured 0.967 instead of 1.0 with a real neural embedder even though
    the extractive generator is grounded by construction."""
    diluting_evidence = [
        FakeEvidence("c3",
            "Nikola Tesla was born in Smiljan. He later moved to the United States. "
            "He worked extensively on wireless power transmission. His rivalry with "
            "Edison became famous. He held over 300 patents by the end of his life."
        ),
    ]
    answer = "Nikola Tesla was born in Smiljan."
    report = check_grounding(answer, diluting_evidence, EMBEDDER)
    assert report.overall_score >= 0.99
    assert report.verdict == "grounded"
    assert report.claims[0].best_evidence_chunk_id == "c3"  # still maps back to the
    # PARENT chunk id, not some internal per-sentence id the caller has no use for


def test_best_evidence_chunk_id_maps_sub_sentence_match_back_to_parent_chunk():
    """A chunk with 3 sentences, only the LAST of which matches the claim --
    confirms per-sentence matching still correctly attributes the match to
    the whole chunk's id, not just when the match happens to be sentence 1."""
    evidence = [
        FakeEvidence("multi-sentence-chunk",
            "Paris is the capital of France. The Eiffel Tower is located there. "
            "The Louvre museum holds the Mona Lisa."
        ),
    ]
    answer = "The Louvre museum holds the Mona Lisa."
    report = check_grounding(answer, evidence, EMBEDDER)
    assert report.claims[0].is_grounded is True
    assert report.claims[0].best_evidence_chunk_id == "multi-sentence-chunk"