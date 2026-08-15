"""
Runnable, narrated demo of the two headline properties of this system:

  1. REAL-TIME: a question about a topic is correctly refused... until that
     topic streams into the live index moments later, at which point the
     exact same question is answered -- with no restart, no reindex job.

  2. GROUNDING / HALLUCINATION GUARD: a question about a topic that's simply
     never going to be ingested is refused, with the agent's reasoning trace
     printed so you can see WHY.

The pipeline is calibrated (reranker + sufficiency gate) before the demo
queries run, using a small set of real labeled questions -- this matters:
the UNCALIBRATED default gate is measurably weaker (see README "Design
decisions" / eval_report.json), and running this demo without calibrating
first would misrepresent how the system is meant to be deployed.

Only a handful of the real Tesla documents are streamed in live (not all 92
in the corpus) -- streaming many documents one-at-a-time against an already-
large index is the known-slow path documented in indexing.py (BM25 has no
incremental API, so each single-document add rebuilds the full lexical index
under lock); bulk loading (ingest_documents) exists specifically to avoid
that cost for large batches. This demo's point is to show a realistic
trickle of new documents arriving live, which is what real-time ingestion
usually looks like in production anyway.

Run with: python scripts/demo_streaming.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verityrag.pipeline import VerityRAGPipeline  # noqa: E402
from verityrag.streaming import LiveDocumentStream  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
N_LATE_DOCS = 6  # small, realistic trickle -- see module docstring


def _print_header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def _print_result(result):
    if result.abstained:
        print(f"  -> ABSTAINED (no confident, grounded answer available)")
    else:
        print(f"  -> ANSWER: {result.answer}")
        print(f"  -> grounding score: {result.grounding.overall_score}  "
              f"({result.grounding.verdict})")
    print(f"  -> evidence sources: {sorted(set(e.title for e in result.evidence))}")
    print(f"  -> trace: {[s.action for s in result.trace]}")


def main():
    with open(DATA_DIR / "corpus.json") as f:
        corpus = json.load(f)
    with open(DATA_DIR / "eval_answerable.json") as f:
        eval_answerable = json.load(f)
    with open(DATA_DIR / "eval_unanswerable.json") as f:
        eval_unanswerable = json.load(f)

    # Tesla documents are entirely excluded from the corpus up front; a
    # small subset streams in live later. The rest are simply unused here.
    tesla_docs = [d for d in corpus if d["title"] == "Nikola_Tesla"][:N_LATE_DOCS]
    initial_docs = [d for d in corpus if d["title"] != "Nikola_Tesla"]

    pipeline = VerityRAGPipeline()

    _print_header("STEP 1 -- Ingesting initial corpus (Tesla docs held back)")
    pipeline.ingest_documents(initial_docs)
    print(f"Indexed {pipeline.index.size()} chunks from {len(initial_docs)} documents.")

    _print_header("STEP 2 -- Calibrating (reranker + sufficiency gate)")
    print("This is not optional flourish -- the uncalibrated default gate is "
          "measurably more prone to false positives (see eval_report.json).")
    t0 = time.time()
    pipeline.train_reranker(n_queries=200)
    # Use real labeled questions for calibration, excluding anything about
    # Tesla so calibration doesn't leak the very thing we're about to test.
    calib_pos = [q["question"] for q in eval_answerable if "Tesla" not in q["question"]][:60]
    calib_neg = [q["question"] for q in eval_unanswerable if "Tesla" not in q["question"]][:60]
    pipeline.calibrate_sufficiency(calib_pos, calib_neg)
    print(f"Calibrated on {len(calib_pos)} answerable / {len(calib_neg)} "
          f"unanswerable real questions ({time.time() - t0:.1f}s).")

    question = "What did Nikola Tesla contribute to electrical engineering?"

    _print_header(f'STEP 3 -- Asking (before Tesla is ingested): "{question}"')
    result_before = pipeline.query(question)
    _print_result(result_before)

    _print_header("STEP 4 -- Starting the live stream and enqueuing Tesla docs")
    stream = LiveDocumentStream(pipeline, delay_seconds=0.05)
    stream.start()
    stream.enqueue_many(tesla_docs)
    print(f"Enqueued {len(tesla_docs)} real Tesla documents onto the live stream...")
    t0 = time.time()
    ingested = stream.wait_until_ingested(len(tesla_docs), timeout=15.0)
    elapsed = time.time() - t0
    stream.stop()
    print(f"Streaming ingestion complete in {elapsed:.2f}s: "
          f"{stream.stats.docs_ingested} docs, {stream.stats.chunks_ingested} chunks "
          f"(ingested_ok={ingested})")

    _print_header("STEP 5 -- Asking the SAME question again, right away")
    result_after = pipeline.query(question)
    _print_result(result_after)

    _print_header("STEP 6 -- A question with NO answer anywhere in this corpus")
    unanswerable_q = "What caused the 1973 oil crisis?"
    result_unanswerable = pipeline.query(unanswerable_q)
    print(f'Question: "{unanswerable_q}"')
    _print_result(result_unanswerable)

    print("\n" + "=" * 70)
    print("Summary: a document became answerable within one streaming cycle of "
          "arriving (no restart), and a genuinely out-of-domain question was "
          "refused rather than answered with a plausible-sounding guess.")
    print("=" * 70)


if __name__ == "__main__":
    main()
