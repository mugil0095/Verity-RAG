from verityrag.embedding import HashingEmbedder
from verityrag.generation import ExtractiveGenerator, LLMGenerator


class FakeEvidence:
    def __init__(self, chunk_id, text):
        self.chunk_id = chunk_id
        self.text = text


EMBEDDER = HashingEmbedder(n_features=4096)
EVIDENCE = [
    FakeEvidence("c1", "The steam engine converts thermal energy into mechanical work. "
                       "It was central to the Industrial Revolution."),
    FakeEvidence("c2", "James Watt significantly improved the efficiency of the steam engine in 1776."),
]


def test_extractive_generator_uses_only_evidence_sentences():
    gen = ExtractiveGenerator(max_sentences=2)
    result = gen.generate("Who improved the steam engine?", EVIDENCE, EMBEDDER)
    # every produced sentence must literally appear in some evidence chunk
    for sentence in result.text.split(". "):
        sentence = sentence.strip().rstrip(".")
        assert any(sentence in ev.text for ev in EVIDENCE if sentence)


def test_extractive_generator_tracks_used_chunk_ids():
    gen = ExtractiveGenerator(max_sentences=2)
    result = gen.generate("steam engine efficiency Watt", EVIDENCE, EMBEDDER)
    assert len(result.used_chunk_ids) > 0
    assert set(result.used_chunk_ids).issubset({"c1", "c2"})


def test_extractive_generator_no_evidence_abstains_gracefully():
    gen = ExtractiveGenerator()
    result = gen.generate("anything", [], EMBEDDER)
    assert "don't have enough" in result.text.lower()
    assert result.used_chunk_ids == []


def test_llm_generator_calls_injected_function_with_evidence_in_prompt():
    captured = {}

    def fake_complete(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Stubbed generative answer."

    gen = LLMGenerator(complete_fn=fake_complete)
    result = gen.generate("Who improved the steam engine?", EVIDENCE, EMBEDDER)

    assert result.text == "Stubbed generative answer."
    assert result.generator == "llm"
    assert "James Watt" in captured["prompt"]  # evidence really was placed in the prompt
