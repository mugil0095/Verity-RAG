[![CI](https://github.com/mugil0095/Verity-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/mugil0095/Verity-RAG/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mugil0095/Verity-RAG/blob/master/LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

# VerityRAG 

**Real-time agentic RAG with grounding & hallucination detection.** [Live Link](https://verityrag.streamlit.app/)

VerityRAG ingests documents continuously (no offline reindex step), answers
questions through a bounded multi-hop retrieval agent, and refuses to answer
rather than hallucinate when it doesn't have grounded evidence.

```
Coverage on real answerable questions:        93.3%
Correctly abstained on held-out-topic Qs:      77.3%   (the hallucination guard)
Query latency:                                 p50 ~58ms / p95 ~88ms
```
Full numbers in [`eval_report.json`](eval_report.json), reproducible with `python -m verityrag.eval`.

## Why this project

Three things RAG teams are actually hiring for: **agentic** multi-step
retrieval, **grounding/hallucination mitigation**, and **RAG evaluation
tooling**. This is a working implementation of all three, with tests.

## Architecture

```mermaid
flowchart TB
    subgraph ingest["Real-Time Ingestion"]
        stream["LiveDocumentStream\n(background thread)"] --> chunk["chunking.py"]
        bulk["bulk loader\n(initial corpus)"] --> chunk
        chunk --> embed["HashingEmbedder\n(stateless, streaming-safe)"]
        embed --> index[("LiveIndex\ndense vectors + BM25")]
    end

    subgraph query["Agentic Query Loop"]
        q["question"] --> retrieve["Hybrid Retrieval\nBM25 + cosine"]
        index -.read.-> retrieve
        retrieve --> rerank["LightGBM Reranker\n(ICT weak supervision)"]
        rerank --> gate{{"Sufficiency Gate\n(calibrated classifier)"}}
        gate -- "insufficient,\nsome signal" --> reformulate["Query Reformulation"]
        reformulate --> retrieve
        gate -- "insufficient,\nno signal" --> abstain1(["ABSTAIN"])
        gate -- sufficient --> generate["Answer Generator\n(extractive / pluggable LLM)"]
        generate --> ground["Grounding Checker\n(claim decomposition)"]
        ground -- "below threshold" --> abstain2(["ABSTAIN"])
        ground -- grounded --> answer(["Answer +\ngrounding report"])
    end
```

| Layer | File | What it does |
|---|---|---|
| Chunking | `chunking.py` | Sentence-aligned, overlapping chunks |
| Embedding | `embedding.py` | Stateless hashed bag-of-n-grams (default) or a real encoder (opt-in) |
| Indexing | `indexing.py`, `elasticsearch_index.py` | Thread-safe incremental dense + BM25 index (pluggable Elasticsearch backend) |
| Retrieval | `retrieval.py` | Hybrid lexical + dense search |
| Reranking | `reranker.py` | LightGBM `LGBMRanker`, ICT weak supervision |
| Sufficiency | `sufficiency.py` | Calibrated classifier: enough evidence to try? |
| Agent | `agent.py`, `agent_reformulate.py` | Bounded retrieve/reformulate/generate/ground/abstain loop |
| Generation | `generation.py`, `llm_providers.py` | Extractive default + pluggable real-LLM generators |
| Grounding | `grounding.py` | Claim decomposition + evidence matching |
| Streaming | `streaming.py` | Background real-time ingestion |
| Serving | `api.py`, `app.py` | FastAPI backend, Streamlit frontend |
| Eval | `eval.py` | Full-corpus evaluation harness |

## Evaluation data

Built from the real **SQuAD 1.1 dev set** — real Wikipedia paragraphs and
real questions, not synthetic text:

- **620 documents / 845 chunks** across 12 topics
- **150 answerable questions** from that corpus
- **150 questions from 36 topics never ingested** — tests whether the
  system hallucinates on things it doesn't know

Calibration and test questions are kept strictly disjoint (50/50 split,
fixed seed) so reported numbers aren't measuring data the classifier was
fit on.

## Design decisions (and the bugs behind them)

Full detail on any of these — including exact numbers, root causes, and
the diagnostic process — is in [ROADMAP.md](ROADMAP.md). Summarized here:

- **No pretrained embedding model by default.** `HashingEmbedder` is a
  stateless hashed bag-of-n-grams transform — a real advantage for
  real-time ingestion, but stopwords swamped the topical signal until
  filtering was added. `SentenceTransformerEmbedder` is an opt-in swap
  behind the same interface (see "Real embeddings, measured").
- **The lexical tokenizer didn't strip punctuation.** Found by comparing
  against Elasticsearch: `rank_bm25`'s tokenizer was pure whitespace
  splitting, so `"Tesla,"` and `"Tesla"` never matched. Fixed to a proper
  regex tokenizer. Real trade-off: guard improved (73.3%→77.3%) but
  coverage dropped slightly (96%→93.3%) — some of the old bug's
  punctuation-attached matches were accidentally helping a few questions
  pass the sufficiency gate.
- **A single relevance threshold doesn't separate answerable from
  unanswerable questions** — their score distributions heavily overlap.
  Fixed with a small LightGBM classifier (`sufficiency.py`) trained on
  multiple retrieval features instead.
- **Query reformulation was amplifying hallucination risk.** The retry
  logic assumed the first pass found something relevant to refine; for
  genuinely out-of-domain questions it pulled the search toward a wrong
  match instead of away. Guard was 9.3% with unconditional reformulation,
  77.3% once gated on the sufficiency classifier's own confidence.
- **Bulk ingestion was accidentally O(n²)** — `rank_bm25` has no
  incremental update API, so adding documents one at a time rebuilt the
  entire lexical index every call (61s for the initial corpus). Bulk
  loading now chunks everything first and rebuilds once: 61s → 2.7s.
  True streaming ingestion still rebuilds per document by design
  (~430ms/doc) — Elasticsearch is the production fix (see below).
- **Real embeddings crashed on Windows, twice** — a `STATUS_ACCESS_VIOLATION`
  from conflicting OpenMP runtimes (PyTorch + MKL-linked scikit-learn/
  LightGBM), fixed at load time; a second crash from thread-pool
  contention under sustained use, fixed with `OMP_NUM_THREADS=1`. Both
  real fixes, but forcing single-threaded execution is part of why real
  embeddings pay a latency tax. A later, real ~15% latency improvement
  (`MKL_THREADING_LAYER=GNU`) is available as an opt-in but not the
  default — the crash it works around needed sustained use to surface,
  and this alternative has less runtime behind it so far.
- **Concurrency was designed in but never actually load-tested** —
  `LiveIndex` uses `threading.RLock()` around reads and writes so
  ingestion and queries can happen at once, reasoned about but not
  verified until `scripts/concurrent_load_check.py` fired genuine
  concurrent HTTP traffic at a real running server: zero errors across
  160+ concurrent ingest/query calls. Holds up because `VectorIndex.add()`
  builds a brand-new array (`np.vstack`) rather than mutating one in
  place, so a reader's snapshot stays valid through a concurrent write.

## Real embeddings, measured

`SentenceTransformerEmbedder` (`all-MiniLM-L6-v2`) is opt-in via
`python -m verityrag.eval --real-embeddings`:

| Metric | HashingEmbedder (default) | SentenceTransformerEmbedder |
|---|---|---|
| Coverage | 93.3% | 92% |
| Hallucination guard | 77.3% | **96%** |
| Keyword hit rate | 67.1% | 76.8% |
| Avg grounding score | 1.0 | 1.0 |
| Latency, p50 | ~58ms | ~2,175ms |

The tokenizer fix above hit this path much harder (guard 82.7%→96% here,
vs. 73.3%→77.3% for the default) because the real-embeddings-trained
sufficiency classifier relies on lexical score as its single most
important feature. Latency is why this isn't the default — real-time
positioning doesn't currently justify the cost. Swap it in explicitly
where the trade-off is worth it.

## Real LLM generation

`ExtractiveGenerator` (default) stitches sentences straight from
evidence — grounded by construction. `LLMGenerator` calls a real model
instead:

```python
from verityrag.pipeline import VerityRAGPipeline
from verityrag.generation import LLMGenerator
from verityrag.llm_providers import gemini_complete_fn  # or anthropic_complete_fn

pipeline = VerityRAGPipeline(generator=LLMGenerator(complete_fn=gemini_complete_fn))
```

| | `anthropic_complete_fn` | `gemini_complete_fn` |
|---|---|---|
| Setup | `pip install anthropic`, `ANTHROPIC_API_KEY` | `pip install google-genai`, `GEMINI_API_KEY` |
| Cost | Small starter credit, then pay-per-token | Genuine free tier, no card needed |
| Default model | `claude-sonnet-5` | `gemini-3.6-flash` |

Confirmed working end-to-end against the live Gemini API. First
small-sample run (n=8+8): hallucination guard **100%**, coverage 50% —
traced directly to a genuine retrieval gap (the gold-answer chunk
genuinely wasn't in the top-6), not an LLM or grounding-checker problem.
The free tier's daily quota is real and restrictive (~20 requests/day),
so `--max-test-questions N` caps how many questions actually get sent,
for an honest partial measurement (`partial_sample` field in the report)
instead of an infeasible full run:

```
python -m verityrag.eval --real-llm gemini --max-test-questions 8
```

Also live in the deployed app: check "Also show a real LLM's answer" on
the Ask tab to see both generators answer the same question side by
side, off the exact same retrieved evidence.

## Swapping in Elasticsearch

`LexicalIndex` (the `rank_bm25`-based default) has no incremental
indexing API — adding one document rebuilds the *entire* lexical index.
`ElasticsearchLexicalIndex` is a drop-in swap with genuine incremental
indexing:

```python
from verityrag.pipeline import VerityRAGPipeline
from verityrag.elasticsearch_index import ElasticsearchLexicalIndex

pipeline = VerityRAGPipeline(lexical_index=ElasticsearchLexicalIndex())
```

Needs `pip install elasticsearch` and a running instance (not bundled —
for local dev, disable security in `elasticsearch.yml` and cap the JVM
heap explicitly rather than trusting auto-sizing).

Measured against a real running instance, not mocks: comparing it
against `rank_bm25` is what surfaced the tokenizer bug above. At this
project's actual corpus size (845 chunks), Elasticsearch is not faster —
206.0ms/doc (`rank_bm25`) vs. 221.7ms/doc (Elasticsearch) — its fixed
per-call overhead (network round-trip + explicit refresh) exceeds
`rank_bm25`'s rebuild cost at this scale. The architectural principle
still holds for a larger corpus; this one hasn't reached the crossover
point yet, reported honestly rather than only measured at a scale picked
to make the swap look good.

## Known limitations

Tracked as an actual backlog in [ROADMAP.md](ROADMAP.md).

- **Extractive generation by default** trades fluent prose for a
  structural guarantee against hallucination — `LLMGenerator` swaps this
  for a real model when you want it.
- **77.3% hallucination-guard rate by default, not 100%** — reported
  honestly. A real embedder closes part of this gap at a real latency
  cost (see above).
- **Streaming throughput** is bounded by the BM25 rebuild cost described
  above — fine for a trickle of documents, not bulk-loading thousands live.
- **The live demo's LLM comparison feature shares one Gemini API key
  across every visitor** — the free tier's daily quota is shared too,
  not per-visitor. Handled gracefully (a clear error on the LLM side,
  extractive answer still works) rather than crashing, but it's a real,
  known constraint of a free-tier key on a public app.

## Running it

```bash
pip install -r requirements.txt
pip install -e .

python data/build_corpus.py       # rebuild the real eval corpus (SQuAD)
pytest tests/ -v                  # run the test suite
python -m verityrag.eval          # full evaluation
python scripts/demo_streaming.py  # narrated real-time + hallucination-guard demo
uvicorn verityrag.api:app --reload
streamlit run app.py              # recommended for demoing
```

### Streamlit frontend

Two tabs: **Ask a question** (load the corpus, calibrate, ask anything —
see the answer, grounding score, and agent trace; optionally, if a
`GEMINI_API_KEY` is configured, compare the extractive and LLM
generators side-by-side) and **Real-time streaming demo** (watch a
question get refused, documents stream in live, the same question get
answered, and a genuinely out-of-domain question stay correctly
refused). Covered by its own `AppTest`-based test suite.

### Minimal usage

```python
from verityrag import VerityRAGPipeline

pipeline = VerityRAGPipeline()
pipeline.ingest_document("d1", "Steam Engine", "The steam engine converts heat into mechanical work...")
pipeline.train_reranker()

result = pipeline.query("How does a steam engine work?")
print(result.answer, result.grounding.overall_score, result.abstained)
```

## License

MIT — see [LICENSE](LICENSE).