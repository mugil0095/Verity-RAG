"""
Answer generation.

`AnswerGenerator` is an interface; two implementations are provided:

  ExtractiveGenerator  -- the default. Selects and stitches the most query-
    relevant sentences straight out of the retrieved evidence. It is "grounded
    by construction" since it literally cannot say anything the evidence
    doesn't contain -- this makes it a safe, always-available default when no
    LLM API is configured (e.g. this sandbox, which has no route to an LLM
    provider), and a useful baseline against which to measure any generative
    model you plug in later.

  LLMGenerator  -- production swap-in. Calls out to a real generative model
    (Claude, GPT, a local model, etc.) via an injected `complete_fn`. Because
    real LLMs *can* hallucinate even with correct evidence in context, this is
    exactly the component the grounding/hallucination detector (grounding.py)
    exists to check -- ExtractiveGenerator answers don't need that check as
    urgently as LLMGenerator answers do, but the pipeline runs the same
    grounding check on both, which is what you want in production: verify the
    output regardless of which generator produced it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .embedding import Embedder, cosine_sim_matrix


@dataclass
class GeneratedAnswer:
    text: str
    generator: str
    used_chunk_ids: list[str]


class AnswerGenerator:
    name: str = "base"

    def generate(self, query: str, evidence: list, embedder: Embedder) -> GeneratedAnswer:
        raise NotImplementedError


class ExtractiveGenerator(AnswerGenerator):
    name = "extractive"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def generate(self, query: str, evidence: list, embedder: Embedder) -> GeneratedAnswer:
        from .chunking import split_sentences  # local import avoids a cycle at module load

        if not evidence:
            return GeneratedAnswer(
                text="I don't have enough grounded evidence in the index to answer this.",
                generator=self.name,
                used_chunk_ids=[],
            )

        candidates: list[tuple[str, str]] = []  # (sentence, chunk_id)
        for chunk in evidence:
            for sent in split_sentences(chunk.text):
                candidates.append((sent, chunk.chunk_id))

        if not candidates:
            return GeneratedAnswer(text="", generator=self.name, used_chunk_ids=[])

        sent_texts = [c[0] for c in candidates]
        query_vec = embedder.embed([query])
        sent_vecs = embedder.embed(sent_texts)
        sims = cosine_sim_matrix(query_vec, sent_vecs)[0]

        ranked_idx = sims.argsort()[::-1][: self.max_sentences]
        ranked_idx = sorted(ranked_idx)  # restore original reading order for coherence

        selected_sentences = [sent_texts[i] for i in ranked_idx]
        used_chunk_ids = list(dict.fromkeys(candidates[i][1] for i in ranked_idx))

        answer_text = " ".join(selected_sentences)
        return GeneratedAnswer(text=answer_text, generator=self.name, used_chunk_ids=used_chunk_ids)


class LLMGenerator(AnswerGenerator):
    """
    Production swap-in. `complete_fn(prompt: str) -> str` should call a real
    model (Anthropic/OpenAI/local). Not invoked anywhere in this build by
    default because no LLM API credential/route is available in this
    environment -- wiring it up is a one-line change at pipeline construction
    time (see pipeline.py / README "Swapping in a real LLM").
    """
    name = "llm"

    def __init__(self, complete_fn: Callable[[str], str]):
        self._complete_fn = complete_fn

    def generate(self, query: str, evidence: list, embedder: Embedder) -> GeneratedAnswer:
        context = "\n\n".join(f"[{e.chunk_id}] {e.text}" for e in evidence)
        prompt = (
            "Answer the question using ONLY the evidence below. "
            "If the evidence is insufficient, say so explicitly.\n\n"
            f"Evidence:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        text = self._complete_fn(prompt)
        return GeneratedAnswer(text=text, generator=self.name,
                                used_chunk_ids=[e.chunk_id for e in evidence])
