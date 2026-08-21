"""
Measures the actual per-document streaming-ingestion cost for the lexical
index -- the second of the two open questions about the Elasticsearch
swap (the first, whether eval numbers change, is what eval.py --elasticsearch
checks). Mirrors how the original ~430ms/doc rank_bm25 number was measured:
build up a large index first via bulk loading, THEN time adding a few MORE
documents ONE AT A TIME -- streaming-style, not bulk -- since that's the
actual real-time ingestion path this project's thesis depends on, and
rank_bm25's full-rebuild cost only shows up once the index it's rebuilding
is already large.

Reports MEDIAN (p50), not mean, matching how every other latency number in
this project is reported (eval.py's latency_p50_ms/p95_ms) -- not an
arbitrary choice here either: initial runs showed roughly 2 out of every
6-8 streamed documents spiking to ~2.5s while the rest held steady around
~220ms, reproducible even with explicit gc.collect() before every call
(ruling out garbage-collection timing as the cause) and with the spike's
position shifting between runs rather than always landing on the same
call (ruling out a simple "first N calls" warm-up effect too). Whatever
the exact cause, a mean is exactly the wrong statistic for a distribution
with occasional large outliers -- median is what the rest of this
project already uses for precisely this reason.

Run with (rank_bm25 baseline only, no Elasticsearch needed):
         python scripts/benchmark_lexical_index.py
Run with the Elasticsearch comparison too (needs a running instance):
         python scripts/benchmark_lexical_index.py --elasticsearch
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verityrag.eval import _load  # noqa: E402
from verityrag.pipeline import VerityRAGPipeline  # noqa: E402


def benchmark(lexical_index=None, n_streamed_docs: int = 16):
    corpus = _load("corpus.json")
    bulk_docs = corpus[:-n_streamed_docs]
    streamed_docs = corpus[-n_streamed_docs:]

    pipeline = VerityRAGPipeline(lexical_index=lexical_index)
    label = type(pipeline.index.lexical_index).__name__

    print(f"[{label}] Bulk-loading {len(bulk_docs)} docs (not timed -- this is the fast path already)...")
    pipeline.ingest_documents(bulk_docs)
    print(f"[{label}] Index now at {pipeline.index.size()} chunks. "
          f"Streaming {len(streamed_docs)} more docs ONE AT A TIME (this is what's timed)...\n")

    per_doc_times = []
    for doc in streamed_docs:
        t0 = time.time()
        pipeline.ingest_document(doc["doc_id"], doc["title"], doc["text"])
        elapsed = time.time() - t0
        per_doc_times.append(elapsed)
        print(f"  {doc['doc_id']}: {elapsed * 1000:.1f}ms  (index now at {pipeline.index.size()} chunks)")

    per_doc_ms = sorted(t * 1000 for t in per_doc_times)
    median_ms = statistics.median(per_doc_ms)
    p95_ms = per_doc_ms[int(len(per_doc_ms) * 0.95) - 1] if len(per_doc_ms) > 1 else per_doc_ms[0]
    print(f"\n[{label}] median: {median_ms:.1f}ms/doc   p95: {p95_ms:.1f}ms/doc   "
          f"(n={len(per_doc_ms)}, against a {pipeline.index.size()}-chunk index)")
    return {"lexical_index": label, "per_doc_ms": per_doc_ms, "median_ms": median_ms, "p95_ms": p95_ms}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--elasticsearch", action="store_true",
                         help="Also benchmark ElasticsearchLexicalIndex (needs a running instance)")
    args = parser.parse_args()

    print("=" * 60)
    print("BASELINE: rank_bm25 (LexicalIndex, current default)")
    print("=" * 60)
    baseline = benchmark(lexical_index=None)

    if args.elasticsearch:
        from verityrag.elasticsearch_index import ElasticsearchLexicalIndex
        print()
        print("=" * 60)
        print("COMPARISON: Elasticsearch (ElasticsearchLexicalIndex)")
        print("=" * 60)
        es_result = benchmark(lexical_index=ElasticsearchLexicalIndex())

        print()
        print("=" * 60)
        print("RESULT")
        print("=" * 60)
        speedup = baseline["median_ms"] / es_result["median_ms"] if es_result["median_ms"] > 0 else float("inf")
        print(f"rank_bm25:     {baseline['median_ms']:.1f}ms/doc median  ({baseline['p95_ms']:.1f}ms p95)")
        print(f"Elasticsearch: {es_result['median_ms']:.1f}ms/doc median  ({es_result['p95_ms']:.1f}ms p95)")
        print(f"Speedup (median): {speedup:.1f}x")
    else:
        print("\nRun with --elasticsearch to compare against a real running instance.")