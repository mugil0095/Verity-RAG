from verityrag.agent import AgentController
from verityrag.chunking import chunk_document
from verityrag.embedding import HashingEmbedder
from verityrag.generation import ExtractiveGenerator
from verityrag.indexing import LiveIndex
from verityrag.sufficiency import SufficiencyGate

DOCS = [
    ("d1", "Tesla", "Nikola Tesla was a Serbian-American inventor and electrical engineer. "
                     "He is best known for his contributions to the design of the modern "
                     "alternating current electricity supply system."),
    ("d2", "Warsaw", "Warsaw is the capital and largest city of Poland. It lies on the "
                     "Vistula river in east-central Poland."),
]


def _build_controller(max_hops=2, sufficiency_threshold=0.15, grounding_abstain_threshold=0.5):
    embedder = HashingEmbedder(n_features=4096)
    index = LiveIndex(embedder)
    for doc_id, title, text in DOCS:
        index.add_chunks(chunk_document(doc_id, title, text, max_tokens=60))
    return AgentController(
        index=index, embedder=embedder, generator=ExtractiveGenerator(),
        max_hops=max_hops, sufficiency_gate=SufficiencyGate(threshold=sufficiency_threshold),
        grounding_abstain_threshold=grounding_abstain_threshold,
    )


def test_agent_answers_confidently_on_in_domain_query():
    agent = _build_controller()
    result = agent.answer("What is Nikola Tesla known for?")
    assert result.abstained is False
    assert result.answer is not None
    assert result.grounding is not None
    assert result.grounding.overall_score >= 0.5


def test_agent_abstains_when_no_relevant_evidence_exists():
    agent = _build_controller(sufficiency_threshold=0.5)  # demanding threshold, tiny 2-doc index
    result = agent.answer("What is the boiling point of liquid helium on Europa?")
    assert result.abstained is True
    assert result.answer is None
    assert result.raw_generated_text is None  # generation never ran -- gate rejected before that


def test_raw_generated_text_preserved_when_grounding_rejects_but_answer_stays_none():
    """The actual point of this field: distinguishing 'gate rejected before
    generation ran' from 'generation ran but grounding rejected it' --
    these need different fixes, and both look identical (answer=None) from
    outside without this. `answer` itself must NOT change behavior -- this
    is a safety-relevant guarantee (an ungrounded answer must never be
    exposed as if it were trusted), so this test locks in that it's still
    None on grounding rejection, exactly as before this field existed."""
    class FakeUngroundedGenerator:
        name = "fake"

        def generate(self, query, evidence, embedder):
            from verityrag.generation import GeneratedAnswer
            return GeneratedAnswer(
                text="A completely unrelated made-up answer with no connection to the evidence.",
                used_chunk_ids=[c.chunk_id for c in evidence],
                generator="fake",
            )

    embedder = HashingEmbedder(n_features=4096)
    index = LiveIndex(embedder)
    for doc_id, title, text in DOCS:
        index.add_chunks(chunk_document(doc_id, title, text, max_tokens=60))
    agent = AgentController(
        index=index, embedder=embedder, generator=FakeUngroundedGenerator(),
        max_hops=2, sufficiency_gate=SufficiencyGate(threshold=0.15),
        grounding_abstain_threshold=0.5,
    )

    result = agent.answer("What is Nikola Tesla known for?")

    assert result.abstained is True
    assert result.answer is None  # unchanged safety behavior -- never expose a rejected answer here
    assert result.raw_generated_text is not None
    assert "unrelated made-up answer" in result.raw_generated_text  # the actual generated text, preserved


def test_agent_abstains_on_empty_index():
    embedder = HashingEmbedder(n_features=2048)
    empty_index = LiveIndex(embedder)
    agent = AgentController(index=empty_index, embedder=embedder, generator=ExtractiveGenerator())
    result = agent.answer("anything at all")
    assert result.abstained is True
    assert result.evidence == []


def test_agent_respects_max_hops_bound():
    agent = _build_controller(max_hops=1, sufficiency_threshold=0.99)  # impossible to satisfy
    result = agent.answer("What is Nikola Tesla known for?")
    # retrieve steps should never exceed max_hops + 1
    retrieve_steps = [s for s in result.trace if s.action == "retrieve"]
    assert len(retrieve_steps) <= 2


def test_agent_trace_records_each_hop():
    agent = _build_controller()
    result = agent.answer("Where is Warsaw located?")
    actions = [s.action for s in result.trace]
    assert "retrieve" in actions
    assert actions[-1] in ("answer", "abstain")


def test_reformulation_does_not_fire_when_hop0_confidently_rejected():
    """Regression test for a real bug found during evaluation: when hop 0 is
    confidently rejected by the sufficiency gate (not just below-threshold,
    but clearly noise), the agent must NOT reformulate -- reformulating from
    noise pulls in unrelated vocabulary and can inflate hop 1's score enough
    to falsely pass the gate (query drift). On the real 620-doc eval corpus
    this cut the hallucination-guard rate from 0.81 (reformulation off) to
    0.09 (reformulation unconditional) before this gate was added -- see
    agent.py AgentController.__init__ docstring."""
    agent = _build_controller()
    agent.min_confidence_to_reformulate = 0.9  # force "confidently reject, don't retry"
    result = agent.answer("What is the boiling point of liquid helium on Europa?")
    reformulate_steps = [s for s in result.trace if s.action == "reformulate"]
    assert reformulate_steps == []
    assert result.abstained is True