"""
Multi-agent loop: planner -> retriever -> verifier, run once per sub-question.

    PlannerAgent    splits a compound question into standalone sub-questions
                    (planner.py). A question it leaves whole goes straight to
                    the single-agent loop, so single questions are unchanged.
    RetrieverAgent  hybrid retrieval + rerank + sufficiency gate + bounded
                    reformulation for ONE sub-question (AgentController.retrieve).
    VerifierAgent   generates from that evidence and checks it: claim-level
                    grounding, then the optional learned guard / answerability
                    gate (AgentController.respond).

The coordinator merges the parts and re-checks grounding on the merged text.
A sub-question that fails its own checks is reported, never silently dropped:
the answer says which part went unanswered, and `parts` carries each part's
status and its retriever/verifier steps for display.

Why it exists: the single loop retrieves once for the whole question, so a
two-part question gets evidence for one part and an answer that is grounded
but incomplete (see compound_eval.py for the measured gap).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .agent import AgentController, AgentResult, AgentStep, Retrieval
from .chunking import split_sentences
from .grounding import check_grounding
from .planner import Plan, Planner
from .retrieval import RetrievedChunk


@dataclass
class PartResult:
    sub_question: str
    answer: str | None
    abstained: bool
    abstain_reason: str | None
    grounding_score: float | None
    evidence_chunk_ids: list[str]
    retriever_steps: list[AgentStep]
    verifier_steps: list[AgentStep]


@dataclass
class MultiAgentResult(AgentResult):
    plan: list[str] = field(default_factory=list)
    parts: list[PartResult] = field(default_factory=list)


class PlannerAgent:
    def __init__(self, planner: Planner | None = None):
        self.planner = planner or Planner()

    def run(self, question: str) -> Plan:
        return self.planner.plan(question)


class RetrieverAgent:
    def __init__(self, controller: AgentController):
        self.controller = controller

    def run(self, sub_question: str) -> Retrieval:
        return self.controller.retrieve(sub_question)


class VerifierAgent:
    def __init__(self, controller: AgentController):
        self.controller = controller

    def run(self, sub_question: str, retrieval: Retrieval) -> AgentResult:
        return self.controller.respond(sub_question, retrieval)


def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen, out = set(), []
    for c in chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            out.append(c)
    return out


def _join_unique(answers: list[str]) -> str:
    """Join part answers, dropping a sentence a previous part already gave
    (two sub-questions often retrieve the same chunk)."""
    seen, out = set(), []
    for answer in answers:
        for sentence in split_sentences(answer):
            if sentence not in seen:
                seen.add(sentence)
                out.append(sentence)
    return " ".join(out)


class MultiAgentController:
    def __init__(self, controller: AgentController, planner: Planner | None = None):
        self.controller = controller
        self.planner = PlannerAgent(planner)
        self.retriever = RetrieverAgent(controller)
        self.verifier = VerifierAgent(controller)

    def answer(self, query: str) -> AgentResult:
        plan = self.planner.run(query)
        if len(plan.sub_questions) < 2:
            return self.controller.answer(query)

        trace = [AgentStep(hop=0, query_used=" | ".join(plan.sub_questions),
                           n_candidates=len(plan.sub_questions), top_score=0.0, action="plan")]
        results: list[AgentResult] = []
        parts: list[PartResult] = []
        for sub in plan.sub_questions:
            retrieval = self.retriever.run(sub)
            n_retrieve_steps = len(retrieval.trace)
            res = self.verifier.run(sub, retrieval)  # appends its steps to the same trace list
            results.append(res)
            parts.append(PartResult(
                sub_question=sub, answer=res.answer, abstained=res.abstained,
                abstain_reason=res.abstain_reason,
                grounding_score=res.grounding.overall_score if res.grounding else None,
                evidence_chunk_ids=[c.chunk_id for c in res.evidence],
                retriever_steps=res.trace[:n_retrieve_steps],
                verifier_steps=res.trace[n_retrieve_steps:],
            ))
            trace.extend(res.trace)
        return self._merge(query, plan, results, parts, trace)

    def _merge(self, query: str, plan: Plan, results: list[AgentResult],
               parts: list[PartResult], trace: list[AgentStep]) -> MultiAgentResult:
        answered = [r for r in results if not r.abstained]
        evidence = _dedupe([c for r in results for c in r.evidence])

        def abstain(reason: str | None, raw: str | None = None, grounding=None) -> MultiAgentResult:
            trace.append(AgentStep(hop=len(trace), query_used=query, n_candidates=0,
                                   top_score=0.0, action="abstain"))
            return MultiAgentResult(query=query, answer=None, grounding=grounding, evidence=evidence,
                                    hops_used=len(trace), abstained=True, trace=trace,
                                    raw_generated_text=raw, abstain_reason=reason,
                                    plan=plan.sub_questions, parts=parts)

        if not answered:
            return abstain(results[0].abstain_reason)

        text = _join_unique([r.answer for r in answered])
        answered_evidence = _dedupe([c for r in answered for c in r.evidence])
        report = check_grounding(text, answered_evidence, self.controller.embedder)
        trace.append(AgentStep(hop=len(trace), query_used=query, n_candidates=len(answered_evidence),
                               top_score=0.0, action="merge"))
        if report.overall_score < self.controller.grounding_abstain_threshold:
            return abstain("grounding", raw=text, grounding=report)

        unanswered = [p.sub_question for p in parts if p.abstained]
        final = text + (" Not answered (insufficient evidence): " + "; ".join(unanswered) if unanswered else "")
        guards = [r.guard_p_hallucinated for r in answered if r.guard_p_hallucinated is not None]
        trace.append(AgentStep(hop=len(trace), query_used=query, n_candidates=0,
                               top_score=0.0, action="answer"))
        return MultiAgentResult(query=query, answer=final, grounding=report, evidence=evidence,
                                hops_used=len(trace), abstained=False, trace=trace,
                                raw_generated_text=text,
                                guard_p_hallucinated=max(guards) if guards else None,
                                plan=plan.sub_questions, parts=parts)
