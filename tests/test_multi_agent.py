from verityrag.agent import AgentController
from verityrag.chunking import chunk_document
from verityrag.embedding import HashingEmbedder
from verityrag.generation import ExtractiveGenerator, GeneratedAnswer
from verityrag.indexing import LiveIndex
from verityrag.multi_agent import MultiAgentController, MultiAgentResult
from verityrag.pipeline import VerityRAGPipeline
from verityrag.sufficiency import SufficiencyGate

DOCS = [
    ("d1", "Tesla", "Nikola Tesla was a Serbian-American inventor and electrical engineer. "
                     "He is best known for his contributions to the design of the modern "
                     "alternating current electricity supply system."),
    ("d2", "Warsaw", "Warsaw is the capital and largest city of Poland. It lies on the "
                     "Vistula river in east-central Poland."),
]
TESLA = "What is Nikola Tesla known for?"
WARSAW = "Where is Warsaw located?"
HELIUM = "What is the boiling point of liquid helium on Europa?"
SWALLOW = "What is the airspeed velocity of an unladen swallow on Mars?"


def _controller(generator=None):
    embedder = HashingEmbedder(n_features=4096)
    index = LiveIndex(embedder)
    for doc_id, title, text in DOCS:
        index.add_chunks(chunk_document(doc_id, title, text, max_tokens=60))
    return AgentController(index=index, embedder=embedder, generator=generator or ExtractiveGenerator(),
                           sufficiency_gate=SufficiencyGate(threshold=0.3))


def _two(first, second):
    return f"{first} Also, {second[0].lower()}{second[1:]}"


def test_retrieve_then_respond_is_the_single_agent_answer():
    agent = _controller()
    for question in (TESLA, HELIUM):
        split = agent.respond(question, agent.retrieve(question))
        whole = agent.answer(question)
        assert (split.abstained, split.abstain_reason, split.answer) == (whole.abstained, whole.abstain_reason, whole.answer)
        assert [c.chunk_id for c in split.evidence] == [c.chunk_id for c in whole.evidence]
        assert [s.action for s in split.trace] == [s.action for s in whole.trace]


def test_single_question_takes_the_single_agent_path():
    agent = _controller()
    result = MultiAgentController(agent).answer(TESLA)
    assert not isinstance(result, MultiAgentResult)
    assert result.answer == agent.answer(TESLA).answer


def test_two_part_question_gets_an_answer_for_each_part():
    result = MultiAgentController(_controller()).answer(_two(TESLA, WARSAW))
    assert isinstance(result, MultiAgentResult)
    assert result.plan == [TESLA, WARSAW]
    assert result.abstained is False
    assert "alternating current" in result.answer and "Poland" in result.answer
    assert {c.doc_id for c in result.evidence} == {"d1", "d2"}
    assert [p.abstained for p in result.parts] == [False, False]
    assert result.grounding is not None and result.grounding.overall_score >= 0.5


def test_merged_answer_does_not_repeat_a_sentence_both_parts_returned():
    result = MultiAgentController(_controller()).answer(_two(TESLA, WARSAW))
    sentences = [s.strip() for s in result.answer.split(". ") if s.strip()]
    assert len(sentences) == len(set(sentences))


def test_unanswerable_part_is_reported_not_dropped():
    result = MultiAgentController(_controller()).answer(_two(TESLA, HELIUM))
    assert result.abstained is False
    assert [p.abstained for p in result.parts] == [False, True]
    assert result.parts[1].abstain_reason == "sufficiency"
    assert "alternating current" in result.answer
    assert f"Not answered (insufficient evidence): {HELIUM}" in result.answer
    # The grounding report covers the answered text only, not the notice.
    assert result.grounding is not None and result.raw_generated_text not in (None, "")
    assert "Not answered" not in result.raw_generated_text


def test_abstains_when_every_part_is_unanswerable():
    result = MultiAgentController(_controller()).answer(_two(HELIUM, SWALLOW))
    assert result.abstained is True
    assert result.answer is None
    assert result.abstain_reason == "sufficiency"
    assert [p.abstained for p in result.parts] == [True, True]


def test_abstains_with_the_grounding_reason_when_every_part_fails_grounding():
    class Ungrounded:
        name = "fake"

        def generate(self, query, evidence, embedder):
            return GeneratedAnswer(text="A completely unrelated made-up answer with no link to the evidence.",
                                   used_chunk_ids=[c.chunk_id for c in evidence], generator="fake")

    result = MultiAgentController(_controller(Ungrounded())).answer(_two(TESLA, WARSAW))
    assert result.abstained is True
    assert result.answer is None
    assert result.abstain_reason == "grounding"


def test_trace_records_plan_each_agent_and_merge():
    result = MultiAgentController(_controller()).answer(_two(TESLA, WARSAW))
    actions = [s.action for s in result.trace]
    assert actions[0] == "plan" and actions[-2:] == ["merge", "answer"]
    for part in result.parts:
        assert part.retriever_steps and all(s.action in ("retrieve", "reformulate") for s in part.retriever_steps)
        assert part.verifier_steps[-1].action == "answer"
        assert part.evidence_chunk_ids


def test_pipeline_uses_the_multi_agent_loop_by_default_and_can_turn_it_off():
    pipeline = VerityRAGPipeline()
    for doc_id, title, text in DOCS:
        pipeline.ingest_document(doc_id, title, text)
    question = _two(TESLA, WARSAW)
    assert isinstance(pipeline.query(question), MultiAgentResult)
    pipeline.multi_agent = False
    assert not isinstance(pipeline.query(question), MultiAgentResult)
    assert not isinstance(VerityRAGPipeline(multi_agent=False).query(question), MultiAgentResult)
