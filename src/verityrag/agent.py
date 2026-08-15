"""
Agent controller: the "agentic" part of the system.

Implements a bounded ReAct-style control loop rather than a single fixed
retrieve-then-generate pass:

    retrieve -> rerank -> is evidence sufficient?
        no  -> reformulate query (pseudo-relevance feedback) -> retrieve again (bounded hops)
        yes -> generate -> check grounding
                   grounding too low -> either retry with reformulated query (if hops remain)
                                         or ABSTAIN (refuse to answer) rather than return
                                         an ungrounded/fabricated-sounding answer
                   grounding OK      -> return answer + grounding report

Abstention is a first-class outcome, not an error path: for a trustworthy
system, "I don't have enough grounded evidence" is a correct answer to a
question outside the knowledge base, and returning it instead of a fluent
guess is the entire point of the grounding layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .agent_reformulate import reformulate_query
from .embedding import Embedder
from .generation import AnswerGenerator, GeneratedAnswer
from .grounding import GroundingReport, check_grounding
from .indexing import LiveIndex
from .reranker import rerank
from .retrieval import RetrievedChunk, hybrid_retrieve
from .sufficiency import SufficiencyGate


@dataclass
class AgentStep:
    hop: int
    query_used: str
    n_candidates: int
    top_score: float
    action: str  # "retrieve" | "reformulate" | "generate" | "abstain" | "answer"


@dataclass
class AgentResult:
    query: str
    answer: str | None
    grounding: GroundingReport | None
    evidence: list[RetrievedChunk]
    hops_used: int
    abstained: bool
    trace: list[AgentStep] = field(default_factory=list)


class AgentController:
    def __init__(
        self,
        index: LiveIndex,
        embedder: Embedder,
        generator: AnswerGenerator,
        reranker_model=None,
        sufficiency_gate: SufficiencyGate | None = None,
        max_hops: int = 2,
        grounding_abstain_threshold: float = 0.5,
        top_k: int = 6,
        min_confidence_to_reformulate: float = 0.3,
    ):
        self.index = index
        self.embedder = embedder
        self.generator = generator
        self.reranker_model = reranker_model
        self.sufficiency_gate = sufficiency_gate or SufficiencyGate()
        self.max_hops = max_hops
        self.grounding_abstain_threshold = grounding_abstain_threshold
        self.top_k = top_k
        # Pseudo-relevance-feedback reformulation (agent_reformulate.py) only
        # helps when hop 0 found a genuine, if incomplete, match to refine.
        # Measured directly (see README "Design decisions"): allowing
        # reformulation whenever hop 0 is merely rejected -- rather than only
        # when the GATE'S OWN confidence score says "plausible but
        # incomplete" -- let query drift turn confidently-rejected,
        # out-of-domain queries into false positives by hop 1, cutting the
        # hallucination-guard rate from 0.81 to 0.09 on a 620-doc real-corpus
        # eval. Gating on the calibrated gate's own score (not a separate
        # hand-picked raw-feature check) keeps this consistent with whatever
        # gate is active, calibrated or not.
        self.min_confidence_to_reformulate = min_confidence_to_reformulate

    def _retrieve_and_rerank(self, query: str) -> list[RetrievedChunk]:
        candidates = hybrid_retrieve(query, self.index, self.embedder, top_k=self.top_k)
        if self.reranker_model is not None and candidates:
            candidates = rerank(query, candidates, self.reranker_model)
        return candidates

    def answer(self, query: str) -> AgentResult:
        trace: list[AgentStep] = []
        current_query = query
        best_candidates: list[RetrievedChunk] = []
        best_top_score = -1.0
        ever_sufficient = False

        for hop in range(self.max_hops + 1):
            candidates = self._retrieve_and_rerank(current_query)
            top_score = candidates[0].dense_score if candidates else 0.0
            trace.append(AgentStep(hop=hop, query_used=current_query,
                                    n_candidates=len(candidates), top_score=top_score,
                                    action="retrieve"))

            if candidates and top_score > best_top_score:
                best_candidates = candidates
                best_top_score = top_score

            # Sufficiency gate: default is a single threshold on raw dense
            # score; swap in a CalibratedSufficiencyGate (sufficiency.py) once
            # labeled calibration data is available -- see pipeline.calibrate_sufficiency().
            sufficient = self.sufficiency_gate.is_sufficient(candidates)
            if sufficient:
                ever_sufficient = True
                break
            if hop == self.max_hops:
                break
            if self.sufficiency_gate.score(candidates) < self.min_confidence_to_reformulate:
                # Hop 0 wasn't just "below the bar" -- the gate is confident
                # there's nothing here. Reformulating from noise drifts
                # further from the truth rather than toward it (see
                # __init__ docstring); abstain now instead.
                break

            current_query = reformulate_query(query, current_query, candidates)
            trace.append(AgentStep(hop=hop, query_used=current_query, n_candidates=0,
                                    top_score=0.0, action="reformulate"))

        # Retrieval never found evidence clearing the sufficiency bar in any hop:
        # abstain now rather than generate an answer grounded in noise. This is
        # the primary defense against out-of-domain questions (see
        # test_agent_abstains_when_no_relevant_evidence_exists).
        if not ever_sufficient:
            trace.append(AgentStep(hop=len(trace), query_used=current_query, n_candidates=0,
                                    top_score=best_top_score, action="abstain"))
            return AgentResult(query=query, answer=None, grounding=None, evidence=best_candidates,
                                hops_used=len(trace), abstained=True, trace=trace)

        generated: GeneratedAnswer = self.generator.generate(query, best_candidates, self.embedder)
        trace.append(AgentStep(hop=len(trace), query_used=current_query,
                                n_candidates=len(best_candidates), top_score=best_candidates[0].hybrid_score,
                                action="generate"))

        report = check_grounding(generated.text, best_candidates, self.embedder)

        if report.overall_score < self.grounding_abstain_threshold:
            trace.append(AgentStep(hop=len(trace), query_used=current_query, n_candidates=0,
                                    top_score=0.0, action="abstain"))
            return AgentResult(query=query, answer=None, grounding=report, evidence=best_candidates,
                                hops_used=len(trace), abstained=True, trace=trace)

        trace.append(AgentStep(hop=len(trace), query_used=current_query, n_candidates=0,
                                top_score=0.0, action="answer"))
        return AgentResult(query=query, answer=generated.text, grounding=report,
                            evidence=best_candidates, hops_used=len(trace),
                            abstained=False, trace=trace)
