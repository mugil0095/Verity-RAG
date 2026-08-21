"""
Evaluation harness -- runs the full pipeline against the real corpus built
from SQuAD (data/build_corpus.py) and reports the metrics that matter for a
trustworthy RAG system.

Methodology note: the labeled answerable/unanswerable questions are split
into a calibration set (used to fit the sufficiency classifier,
sufficiency.py) and a disjoint test set (used only for the reported metrics
below). Reporting numbers on the same examples used to fit the classifier
would overstate performance -- this split exists specifically to avoid that.

Metrics:
  - coverage_rate: of real, answerable TEST questions, fraction attempted
    (not wrongly abstained)
  - keyword_hit_rate: of attempted answers, fraction containing the gold
    answer text (a rough, automatic correctness proxy -- not a substitute
    for human/LLM-graded eval, but a real, zero-dependency signal)
  - avg_grounding_score: average faithfulness score on attempted answers
  - hallucination_guard_rate: of TEST questions from topics never ingested,
    fraction correctly abstained rather than answered -- the headline
    hallucination-guard metric
  - latency p50/p95

Run with: python -m verityrag.eval
Run with real neural embeddings instead of the default hashed ones:
         python -m verityrag.eval --real-embeddings
         (needs `pip install sentence-transformers` + internet access for
         the one-time model download -- see embedding.py)
Run with a real LLM generator instead of the default extractive one:
         python -m verityrag.eval --real-llm anthropic
         (no ongoing free tier -- small starter credit only; needs
         `pip install anthropic` + ANTHROPIC_API_KEY -- see llm_providers.py)
         python -m verityrag.eval --real-llm gemini
         (genuine free tier, no card required; needs
         `pip install google-genai` + GEMINI_API_KEY -- see llm_providers.py)
Both together:
         python -m verityrag.eval --real-embeddings --real-llm gemini
Cap how many test questions get sent to the LLM (per class -- e.g. 8 means
at most 8 answerable + 8 unanswerable, not 8 total), for when a real
daily API quota makes the full ~150-question test set infeasible in one
day (measured directly in practice: 20 requests/day for a brand-new
free-tier model on a real account):
         python -m verityrag.eval --real-llm gemini --max-test-questions 8
"""
from __future__ import annotations

import os
# Must happen before ANY of this module's own imports below, not just before
# SentenceTransformerEmbedder gets constructed later -- `from .pipeline import
# VerityRAGPipeline` a few lines down already pulls in LightGBM (via
# reranker.py), and LightGBM's native thread pool needs to see this at ITS
# OWN import/init time, not whenever the embedder happens to be built.
#
# Two real, reproducible Windows crashes led here, in this order:
#  1. STATUS_ACCESS_VIOLATION (0xC0000005) the moment sentence-transformers/
#     torch loaded, because MKL-linked numpy/scikit-learn/LightGBM and torch
#     each bundle their own OpenMP runtime, and loading two into one process
#     can abort outright rather than warn. KMP_DUPLICATE_LIB_OK=TRUE is
#     PyTorch's own documented workaround for that.
#  2. A second, similar-looking crash that only showed up ~100+ questions
#     into the real eval's query loop -- which alternates torch (embed) and
#     LightGBM (rerank) calls on every single question -- but never showed
#     up in isolated testing that used torch alone. That pattern points to
#     runtime thread-pool contention BETWEEN the two OpenMP runtimes, not
#     just the load-time conflict #1 already covers. Forcing single-threaded
#     BLAS/OpenMP execution removes the contention rather than racing it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import random
import statistics
import time
from pathlib import Path

from .embedding import Embedder
from .generation import AnswerGenerator
from .indexing import LexicalIndex
from .pipeline import VerityRAGPipeline

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _load(name: str):
    with open(DATA_DIR / name) as f:
        return json.load(f)


def _split(items: list, calib_fraction: float, seed: int) -> tuple[list, list]:
    items = list(items)
    random.Random(seed).shuffle(items)
    n_calib = max(1, int(len(items) * calib_fraction))
    return items[:n_calib], items[n_calib:]


def run_eval(use_reranker: bool = True, calibrate: bool = True, verbose: bool = True,
             embedder: Embedder | None = None, generator: AnswerGenerator | None = None,
             lexical_index: LexicalIndex | None = None,
             max_test_questions: int | None = None) -> dict:
    """`embedder`: defaults to None, which lets VerityRAGPipeline use its own
    default (HashingEmbedder -- fast, no dependencies, no network). Pass an
    explicit embedder (e.g. SentenceTransformerEmbedder()) to evaluate real
    embedding quality instead -- this is the intended way to measure that
    upgrade's actual impact on coverage/hallucination-guard, rather than
    changing the pipeline's global default (see ROADMAP.md).

    `generator`: same idea, for the answer generator. Defaults to None,
    which lets VerityRAGPipeline use ExtractiveGenerator (grounded by
    construction -- can't hallucinate). Pass an LLMGenerator (see
    generation.py, llm_providers.py) to measure a REAL generator's actual
    hallucination-guard/grounding behavior instead of the structurally-safe
    default -- this is the only way to see the grounding checker catch a
    genuine hallucination rather than a synthetic unit-test one.

    `lexical_index`: same idea again, for the lexical (BM25) index.
    Defaults to None, which lets LiveIndex use the rank_bm25-based
    LexicalIndex (indexing.py) -- no incremental indexing API, full
    rebuild on every add. Pass an ElasticsearchLexicalIndex
    (elasticsearch_index.py) to measure whether swapping it in changes
    any eval numbers at all (it shouldn't -- both are BM25-family scoring
    -- but "shouldn't" isn't "confirmed" until this is actually run).
    Needs a real, running Elasticsearch instance -- see README "Swapping
    in Elasticsearch".

    `max_test_questions`: caps how many TEST questions get queried, per
    class (answerable/unanswerable) -- e.g. 8 means at most 8 answerable +
    8 unanswerable, 16 total, not 8 total. Only affects the measurement
    loop -- calibration is untouched and uses its own full split regardless,
    since calibrate_sufficiency() never calls the generator at all (only
    hybrid_retrieve + feature extraction), so it costs nothing against an
    LLM API quota either way. Added specifically because a real free-tier
    daily quota measured directly in practice (20 requests/day for a
    brand-new model on a real account) makes the full ~150-question test
    set genuinely infeasible in one day -- this is how to get a real,
    honestly-labeled PARTIAL measurement instead of no measurement at all.
    A smaller sample is less statistically precise than the full set, and
    the report says so explicitly when this is used (see partial_sample
    field below) -- this is not a substitute for the full numbers, it's
    the realistic option a hard quota actually leaves available."""
    corpus = _load("corpus.json")
    eval_answerable = _load("eval_answerable.json")
    eval_unanswerable = _load("eval_unanswerable.json")

    calib_pos, test_pos = _split(eval_answerable, calib_fraction=0.5, seed=13)
    calib_neg, test_neg = _split(eval_unanswerable, calib_fraction=0.5, seed=13)

    full_test_pos_size, full_test_neg_size = len(test_pos), len(test_neg)
    if max_test_questions is not None:
        test_pos = test_pos[:max_test_questions]
        test_neg = test_neg[:max_test_questions]

    pipeline = VerityRAGPipeline(embedder=embedder, generator=generator, lexical_index=lexical_index)

    t0 = time.time()
    pipeline.ingest_documents(corpus)
    ingest_time = time.time() - t0
    if verbose:
        print(f"Ingested {len(corpus)} docs -> {pipeline.index.size()} chunks in {ingest_time:.2f}s")

    if use_reranker:
        t0 = time.time()
        trained = pipeline.train_reranker(n_queries=300)
        if verbose:
            print(f"Reranker trained: {trained} ({time.time() - t0:.2f}s)")

    calibrated = False
    if calibrate:
        t0 = time.time()
        calibrated = pipeline.calibrate_sufficiency(
            answerable_questions=[q["question"] for q in calib_pos],
            unanswerable_questions=[q["question"] for q in calib_neg],
        )
        if verbose:
            print(f"Sufficiency gate calibrated: {calibrated} "
                  f"(on {len(calib_pos)} pos / {len(calib_neg)} neg calibration examples, "
                  f"{time.time() - t0:.2f}s)")

    # ---- answerable TEST questions (disjoint from calibration set) ----
    attempted, wrongly_abstained, keyword_hits, grounding_scores, latencies = 0, 0, 0, [], []
    for item in test_pos:
        t0 = time.time()
        result = pipeline.query(item["question"])
        latencies.append(time.time() - t0)
        if result.abstained:
            wrongly_abstained += 1
            continue
        attempted += 1
        grounding_scores.append(result.grounding.overall_score)
        if item["gold_answer"].lower() in (result.answer or "").lower():
            keyword_hits += 1

    # ---- held-out (unanswerable) TEST questions ----
    correctly_abstained = 0
    for item in test_neg:
        result = pipeline.query(item["question"])
        if result.abstained:
            correctly_abstained += 1

    report = {
        "embedder": type(pipeline.embedder).__name__,
        "generator": pipeline.generator.name,
        "lexical_index": type(pipeline.index.lexical_index).__name__,
        "corpus_docs": len(corpus),
        "corpus_chunks": pipeline.index.size(),
        "ingest_time_sec": round(ingest_time, 3),
        "sufficiency_gate_calibrated": calibrated,
        "calibration_examples": {"answerable": len(calib_pos), "unanswerable": len(calib_neg)},
        "test_set": {"answerable": len(test_pos), "unanswerable": len(test_neg)},
        "partial_sample": (
            {"full_test_set_size": {"answerable": full_test_pos_size, "unanswerable": full_test_neg_size},
             "note": "Numbers below are from a SMALL SAMPLE, not the full test set -- "
                     "less statistically precise, not directly comparable to full-set results "
                     "from a different embedder/generator run."}
            if max_test_questions is not None else None
        ),
        "attempted": attempted,
        "wrongly_abstained": wrongly_abstained,
        "coverage_rate": round(attempted / len(test_pos), 3) if test_pos else None,
        "keyword_hit_rate_on_attempted": round(keyword_hits / attempted, 3) if attempted else 0.0,
        "avg_grounding_score_on_attempted": round(statistics.mean(grounding_scores), 3) if grounding_scores else 0.0,
        "correctly_abstained_on_unanswerable": correctly_abstained,
        "hallucination_guard_rate": round(correctly_abstained / len(test_neg), 3) if test_neg else None,
        "latency_p50_ms": round(statistics.median(latencies) * 1000, 1) if latencies else None,
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1] * 1000, 1) if latencies else None,
    }

    if verbose:
        print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-embeddings", action="store_true",
        help="Use SentenceTransformerEmbedder instead of the default HashingEmbedder "
             "(needs `pip install sentence-transformers` + internet for the model download)",
    )
    parser.add_argument(
        "--real-llm", choices=["anthropic", "gemini"], default=None,
        help="Use a real LLM API call instead of the default extractive generator. "
             "'anthropic': needs `pip install anthropic` + ANTHROPIC_API_KEY (no ongoing "
             "free tier, small starter credit only). 'gemini': needs `pip install "
             "google-genai` + GEMINI_API_KEY (genuine free tier, no card required -- "
             "see llm_providers.py). Either makes one real API call per query.",
    )
    parser.add_argument(
        "--max-test-questions", type=int, default=None,
        help="Cap test questions per class (answerable/unanswerable) -- e.g. 8 means "
             "at most 8+8=16 total LLM calls, not the full ~150. For when a real daily "
             "API quota (e.g. a free-tier Gemini limit) makes the full test set "
             "infeasible in one day. Only affects --real-llm runs in practice -- "
             "the default extractive generator has no API cost to limit.",
    )
    parser.add_argument(
        "--elasticsearch", action="store_true",
        help="Use ElasticsearchLexicalIndex instead of the default rank_bm25-based "
             "LexicalIndex (needs `pip install elasticsearch` + a running Elasticsearch "
             "instance -- see README 'Swapping in Elasticsearch'). Checks whether the "
             "swap changes any eval numbers (it shouldn't -- BM25-family scoring "
             "either way).",
    )
    args = parser.parse_args()

    chosen_embedder = None
    if args.real_embeddings:
        from .embedding import SentenceTransformerEmbedder
        chosen_embedder = SentenceTransformerEmbedder()

    chosen_lexical_index = None
    if args.elasticsearch:
        from .elasticsearch_index import ElasticsearchLexicalIndex
        chosen_lexical_index = ElasticsearchLexicalIndex()

    chosen_generator = None
    if args.real_llm == "anthropic":
        from .generation import LLMGenerator
        from .llm_providers import anthropic_complete_fn
        chosen_generator = LLMGenerator(complete_fn=anthropic_complete_fn)
    elif args.real_llm == "gemini":
        from .generation import LLMGenerator
        from .llm_providers import gemini_complete_fn
        chosen_generator = LLMGenerator(complete_fn=gemini_complete_fn)

    run_eval(embedder=chosen_embedder, generator=chosen_generator, lexical_index=chosen_lexical_index,
             max_test_questions=args.max_test_questions)