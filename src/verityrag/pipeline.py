"""
VerityRAGPipeline: the single object that wires every layer together.

    ingest_document()      -> chunk -> embed -> add to live index (real-time,
                               one-doc-at-a-time path, used by streaming.py)
    ingest_documents()     -> BULK loading path: chunks+embeds every doc first,
                               then updates the index ONCE. Deliberately a
                               different code path from ingest_document() --
                               see the docstring on that method for why.
    train_reranker()       -> weak-supervision LightGBM training over current index
    calibrate_sufficiency()-> trains the retrieval-confidence classifier
                               (sufficiency.py) on labeled answerable/should-
                               abstain examples
    query()                -> runs the agent controller and returns an AgentResult
"""
from __future__ import annotations

from .agent import AgentController, AgentResult
from .chunking import chunk_document
from .embedding import Embedder, HashingEmbedder
from .generation import AnswerGenerator, ExtractiveGenerator
from .indexing import LiveIndex
from .reranker import generate_ict_training_data, train_reranker
from .retrieval import hybrid_retrieve
from .sufficiency import CalibrationExample, extract_features, train_sufficiency_gate


class VerityRAGPipeline:
    def __init__(
        self,
        embedder: Embedder | None = None,
        generator: AnswerGenerator | None = None,
        max_hops: int = 2,
    ):
        self.embedder = embedder or HashingEmbedder()
        self.index = LiveIndex(self.embedder)
        self.generator = generator or ExtractiveGenerator()
        self._reranker_model = None
        self.agent = AgentController(
            index=self.index,
            embedder=self.embedder,
            generator=self.generator,
            reranker_model=None,
            max_hops=max_hops,
        )

    def ingest_document(self, doc_id: str, title: str, text: str) -> int:
        """Real-time ingestion path: chunk -> embed -> add to live index
        immediately (one BM25 rebuild per call). This is the path
        LiveDocumentStream uses for genuine one-at-a-time real-time arrivals,
        where per-document freshness matters more than raw throughput."""
        chunks = chunk_document(doc_id, title, text)
        return self.index.add_chunks(chunks)

    def ingest_documents(self, docs: list[dict]) -> int:
        """Bulk loading path (e.g. initial corpus load). Chunks every document
        first, then adds them to the index in ONE call, so the lexical index
        is rebuilt once instead of once per document -- ingest_document()
        calling index.add_chunks() per-doc in a loop is O(n^2) in corpus size
        because BM25Okapi has no incremental API (see indexing.py), which
        matters once you're loading hundreds of documents rather than a
        handful. Real-time single-document arrivals still go through
        ingest_document(), which intentionally keeps the per-call rebuild
        since streamed documents need to be searchable immediately, not
        batched."""
        all_chunks = []
        for d in docs:
            all_chunks.extend(chunk_document(d["doc_id"], d["title"], d["text"]))
        return self.index.add_chunks(all_chunks)

    def train_reranker(self, n_queries: int = 150) -> bool:
        """Trains the LightGBM reranker via ICT weak supervision over whatever
        is currently indexed. Returns False if the index is too small to train on."""
        snapshot = self.index.snapshot()
        if len(snapshot) < 6:
            return False
        examples = generate_ict_training_data(snapshot, self.embedder, n_queries=n_queries)
        if not examples:
            return False
        self._reranker_model = train_reranker(examples)
        self.agent.reranker_model = self._reranker_model
        return True

    def calibrate_sufficiency(
        self,
        answerable_questions: list[str],
        unanswerable_questions: list[str],
    ) -> bool:
        """Trains the multi-feature sufficiency classifier (sufficiency.py) on
        labeled examples: real questions the index SHOULD be able to answer,
        and real questions it should NOT (out-of-domain / not-yet-ingested
        topics). Replaces the default single-threshold gate on
        self.agent.sufficiency_gate. Returns False if too few examples or
        only one class is represented."""
        examples: list[CalibrationExample] = []
        for q in answerable_questions:
            candidates = hybrid_retrieve(q, self.index, self.embedder, top_k=6)
            if self.agent.reranker_model is not None and candidates:
                from .reranker import rerank
                candidates = rerank(q, candidates, self.agent.reranker_model)
            examples.append(CalibrationExample(features=extract_features(candidates), label=1))
        for q in unanswerable_questions:
            candidates = hybrid_retrieve(q, self.index, self.embedder, top_k=6)
            if self.agent.reranker_model is not None and candidates:
                from .reranker import rerank
                candidates = rerank(q, candidates, self.agent.reranker_model)
            examples.append(CalibrationExample(features=extract_features(candidates), label=0))

        try:
            gate = train_sufficiency_gate(examples)
        except ValueError:
            return False
        self.agent.sufficiency_gate = gate
        return True

    def query(self, question: str) -> AgentResult:
        return self.agent.answer(question)
