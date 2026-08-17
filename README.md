[![CI](https://github.com/mugil0095/Verity-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/mugil0095/Verity-RAG/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mugil0095/Verity-RAG/blob/master/LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

# VerityRAG

**Real-time agentic RAG with grounding & hallucination detection.**

# VerityRAG

**Real-time agentic RAG with grounding & hallucination detection.**

VerityRAG ingests documents continuously (no offline reindex step), answers
questions through a bounded multi-hop retrieval agent, and refuses to answer
rather than hallucinate when it doesn't have grounded evidence — verified on
a real corpus, not a toy example.

```
Coverage on real answerable questions:        96.0%
Correctly abstained on held-out-topic Qs:      73.3%   (the hallucination guard)
Query latency:                                 p50 ~150ms / p95 ~400ms
```
Full numbers in [`eval_report.json`](eval_report.json), reproducible with `python -m verityrag.eval`.

## Why this project

Three of the things RAG teams are actually hiring for right now: **agentic**
multi-step retrieval (not single-shot retrieve-then-generate), **grounding /
hallucination mitigation** (the biggest trust problem in production RAG), and
**RAG evaluation tooling** (RAGAS/TruLens-style faithfulness scoring). This
project is a working implementation of all three, end to end, with tests.

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
        gate -- "insufficient,\nsome signal" --> reformulate["Query Reformulation\n(pseudo-relevance feedback)"]
        reformulate --> retrieve
        gate -- "insufficient,\nno signal" --> abstain1(["ABSTAIN"])
        gate -- sufficient --> generate["Answer Generator\n(extractive / pluggable LLM)"]
        generate --> ground["Grounding Checker\n(claim decomposition +\nevidence matching)"]
        ground -- "below threshold" --> abstain2(["ABSTAIN"])
        ground -- grounded --> answer(["Answer +\ngrounding report"])
    end
```

| Layer | File | What it does |
|---|---|---|
| Chunking | `chunking.py` | Sentence-aligned, overlapping chunks |
| Embedding | `embedding.py` | Stateless hashed bag-of-n-grams (no model download needed — see below) |
| Indexing | `indexing.py` | Thread-safe incremental dense + BM25 index |
| Retrieval | `retrieval.py` | Hybrid lexical + dense search |
| Reranking | `reranker.py` | LightGBM `LGBMRanker`, trained via Inverse Cloze Task weak supervision |
| Sufficiency | `sufficiency.py` | Calibrated classifier deciding "do we have enough to even try" |
| Agent | `agent.py`, `agent_reformulate.py` | Bounded retrieve/reformulate/generate/ground/abstain loop |
| Generation | `generation.py` | Extractive default + pluggable `LLMGenerator` interface |
| Grounding | `grounding.py` | Claim decomposition + semantic/lexical evidence matching |
| Streaming | `streaming.py` | Background real-time ingestion |
| Serving | `api.py` | FastAPI `/ingest`, `/query`, `/health`, `/stats` |
| Frontend | `app.py` | Streamlit UI: ask questions, run the live-streaming demo |
| Eval | `eval.py` | Full-corpus evaluation harness |

## Evaluation data

Built from the real **SQuAD 1.1 dev set** (`data/build_corpus.py`) — real
Wikipedia paragraphs and real human-written questions, not synthetic text:

- **620 documents / 845 chunks** across 12 topics, ingested as the corpus
- **150 real questions** answerable from that corpus (`eval_answerable.json`)
- **150 real questions from 36 different topics never ingested**
  (`eval_unanswerable.json`) — used specifically to test whether the system
  hallucinates on things it doesn't know

Calibration and reported-metric questions are kept strictly disjoint (a
50/50 split with a fixed seed) so the reported numbers aren't measuring the
same data the sufficiency classifier was fit on.

## Design decisions (and the bugs that shaped them)

This section exists because the debugging process is the most interesting
part of the project, and I'd rather show real engineering iteration than
pretend it worked perfectly the first time.

**No pretrained neural embedding model.** This was built in a sandboxed
environment with no route to the HuggingFace Hub or any embedding API, so
`embedding.py` uses `sklearn.HashingVectorizer` — a stateless, hash-based
bag-of-n-grams transform — instead of sentence-transformers. That
statelessness turned out to be a genuine advantage for real-time ingestion
(no fitting/refitting step, ever), but it has a real weakness: without IDF
weighting, common words dominated short texts and swamped the topical
signal. Measured directly: an off-topic query scored *higher* raw cosine
similarity than an on-topic one until stopword filtering was added. Fixed by
filtering stopwords in the vectorizer. **Production fix:** swap
`HashingEmbedder` for a real dense encoder behind the same `.embed()`
interface — the rest of the system doesn't care how the vector was produced.

**A single relevance threshold doesn't separate answerable from
unanswerable questions.** The first version gated the agent's "do I have
enough evidence" decision on one number (top-1 cosine similarity). Measured
on the real corpus: on-topic and off-topic-but-coincidentally-keyword-
overlapping questions had heavily overlapping score distributions (0.075–0.574
vs. 0.040–0.221) — no single threshold cleanly separates them. Fixed by
training a small LightGBM classifier (`sufficiency.py`) on multiple retrieval
features (top-1 and top-3 dense scores, raw BM25 score, score gap, candidate
count above a floor) instead of hand-picking one number.

**Query reformulation was amplifying hallucination risk, not reducing it.**
The biggest bug, and the most instructive one. The agent's retry logic
(pseudo-relevance feedback: expand the query using terms from the best
candidate so far) is a real, standard IR technique — but it assumes the
first-pass retrieval found *something* relevant to refine. For genuinely
out-of-domain questions, the "best candidate" is just noise, and expanding
the query with its vocabulary pulled the search *toward* that wrong match
instead of away from it, letting a confidently-wrong answer pass the
sufficiency gate on the second hop. Measured on the full corpus: the
hallucination-guard rate was **9.3%** with unconditional reformulation and
**73.3%** once reformulation was gated on the sufficiency classifier's own
calibrated confidence (only retry when hop 0 looks *plausible but
incomplete*, not when it's already confidently rejected). Locked in with a
regression test (`test_reformulation_does_not_fire_when_hop0_confidently_rejected`).

**Bulk ingestion was accidentally O(n²).** `rank_bm25` has no incremental
update API, so adding documents one at a time — correct for genuine
real-time streaming — meant rebuilding the *entire* lexical index on every
single call. Loading the initial 620-document corpus that way took 61
seconds. Fixed by giving bulk loading (`ingest_documents`) its own code path
that chunks everything first and rebuilds the lexical index exactly once:
**61s → 2.7s.** The true one-at-a-time streaming path (`ingest_document`,
used by `LiveDocumentStream`) still rebuilds per call by design — streamed
documents need to be searchable immediately — and this has a real, measured
cost: streaming 6 individual documents into an already-732-chunk index took
~2.9s (~430ms/doc) versus ~4ms/doc in the bulk path. **Production fix:**
swap the lexical index for Elasticsearch/OpenSearch, which supports true
incremental indexing (this is the same near-real-time refresh model
Elasticsearch itself uses).

**A real neural embedder crashed on Windows — twice, in two different
ways.** Loading `sentence-transformers` (which pulls in PyTorch) into the
same process as scikit-learn and LightGBM (both MKL-linked) triggered a
`STATUS_ACCESS_VIOLATION` (`0xC0000005`) the moment the model loaded — each
library bundles its own OpenMP runtime, and loading more than one into a
process can abort outright on Windows rather than just warn.
`KMP_DUPLICATE_LIB_OK=TRUE` (PyTorch's own documented workaround) fixed
that one. A second, similar-looking crash then showed up over 100
questions into a real eval run, but never in isolated testing that used
the embedder alone — consistent with runtime thread-pool contention between
the two OpenMP runtimes under sustained, tightly-interleaved use (the query
loop alternates torch and LightGBM calls on every single question), rather
than the one-time load conflict the first fix addressed. Forcing
single-threaded execution (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`)
resolved it. Both are real, reproducible fixes for a real class of
Windows-specific issue, not project-specific hacks — but they're also why
real embeddings currently pay a latency tax beyond what's inherent to CPU
transformer inference alone (see "Real embeddings, measured" below).

## Real embeddings, measured

The embedding-layer limitation described above was fixed and measured, not
just predicted. `SentenceTransformerEmbedder` (`embedding.py`) is a real,
working `all-MiniLM-L6-v2` encoder behind the same `Embedder` interface —
opt-in via `python -m verityrag.eval --real-embeddings`, deliberately not
the pipeline's default (see below for why).

| Metric | HashingEmbedder (default) | SentenceTransformerEmbedder |
|---|---|---|
| Coverage | 96% | 89.3% |
| Hallucination guard | 73.3% | **82.7%** |
| Keyword hit rate | 68.1% | 70.1% |
| Avg grounding score | 1.0 | **1.0** |
| Latency, p50 | ~50ms | ~1,764ms |

Not a clean win, reported as measured rather than cherry-picked. The
headline hypothesis is confirmed — a real encoder closes a meaningful chunk
of the hallucination-guard gap (+9.4pp) — but it comes with two real costs:
coverage drops ~7 points (5 more genuinely answerable questions get wrongly
refused), and latency increases roughly 35x. Some of that latency is
inflated by a stability workaround (see "Design decisions" above), and some
by a later fix that embeds evidence at sentence granularity instead of
whole chunks (see below) — more, smaller items through the encoder per
query — rather than being purely inherent to using a real encoder. Even
accounting for both, CPU transformer inference is never going to match
instant hashing.

Avg grounding score was originally 0.967, not 1.0, despite the extractive
generator being grounded by construction. Traced to a real bug, not a
tuning problem: `check_grounding` compared each claim against whole
multi-sentence evidence chunks, and embedding a multi-sentence chunk as one
vector dilutes a claim's similarity to the one sentence it actually
matches — measured directly (no real encoder needed to confirm this part):
a claim scored 1.0 against its own source sentence in isolation but only
0.41 against a 5-sentence chunk containing that sentence plus four
unrelated ones. `HashingEmbedder` never showed this, because its diluted
score (0.41) still cleared the grounding threshold comfortably; a real
encoder's un-diluted score for a genuine match starts from a more moderate
baseline, so the same dilution could tip a genuinely-grounded claim below
threshold. Fixed by matching against individual evidence sentences instead
of whole chunks, and confirmed against the real encoder: **1.0**, exactly
matching `HashingEmbedder`'s ceiling. That same fix moved coverage and
guard by one question each in opposite directions (96→88 became 88→89.3;
84 became 82.7) — the grounding-based final-abstention check in the agent
loop was occasionally tripped by the dilution noise in both directions,
correctly for the wrong reason on one side, incorrectly on the other.

Best current explanation for the remaining coverage gap: SQuAD questions
are often near-paraphrases of their source sentence, sharing exact
vocabulary — precisely what the hash-based lexical matching is good at. A
neural encoder captures broader meaning but can occasionally underweight
that literal overlap for a correct-but-differently-phrased passage.

This is why `SentenceTransformerEmbedder` is not the pipeline default: for
a system explicitly positioned as *real-time*, a >10x latency cost isn't
currently justified by the guard improvement, especially with the
underlying crash issue only worked around (see below), not resolved
cleanly. That's a deliberate, measured decision, not an oversight — swap it
in explicitly wherever the trade-off is worth it for your use case.

## Known limitations

Tracked as an actual backlog in [ROADMAP.md](ROADMAP.md), not just prose here.

- **Extractive generation by default.** The default generator selects and
  stitches sentences straight from retrieved evidence rather than writing
  fluent prose — it's "grounded by construction" and needs no external API,
  which is why `avg_grounding_score` is 1.0 in the eval report. The
  `keyword_hit_rate` of 68% (whether the literal gold-answer text appears
  in the response) reflects this: a real LLM behind the `LLMGenerator`
  interface would likely score higher on precision at the cost of needing
  the grounding checker to do real work (LLMs *can* hallucinate even with
  correct context — extractive generation structurally can't).
- **73.3% hallucination-guard rate on the default configuration, not 100%.**
  Reported honestly rather than tuned to look better. Confirmed, not just
  predicted, that a real neural embedder closes a meaningful part of this
  gap — see "Real embeddings, measured" above — at a real latency cost
  that's exactly why it isn't the default yet.
- **One-at-a-time streaming throughput** is bounded by the BM25 rebuild
  cost described above; fine for a trickle of new documents, not for
  bulk-loading thousands of documents live.

## Running it

```bash
pip install -r requirements.txt
pip install -e .

# Rebuild the real evaluation corpus (fetches the SQuAD dev set)
python data/build_corpus.py

# Run the test suite (58 tests)
pytest tests/ -v

# Run the full evaluation against the real 620-doc corpus
python -m verityrag.eval

# Narrated demo: real-time streaming + hallucination guard, end to end
python scripts/demo_streaming.py

# Run the API
uvicorn verityrag.api:app --reload

# Run the Streamlit frontend (recommended for demoing this project)
streamlit run app.py
```

### Streamlit frontend

`app.py` gives you a real UI over the same pipeline that powers the API —
useful for actually demoing this rather than reading code. Two tabs:

- **Ask a question** — load the real corpus (sidebar), optionally calibrate
  the sufficiency gate, then ask anything; see the answer, grounding score,
  claim-level grounding breakdown, agent trace, and evidence used.
- **Real-time streaming demo** — a one-click version of
  `scripts/demo_streaming.py`: watch a question get refused, watch documents
  stream into the live index with a real progress bar (backed by the actual
  `LiveDocumentStream` background thread, not a fake animation), then watch
  the same question get answered — plus a genuinely out-of-domain question
  that stays correctly refused throughout.

Covered by its own test suite (`tests/test_app.py`) using Streamlit's
`AppTest` framework — same TDD approach as the rest of the project, not just
manual clicking.

### Minimal usage

```python
from verityrag import VerityRAGPipeline

pipeline = VerityRAGPipeline()
pipeline.ingest_document("d1", "Steam Engine", "The steam engine converts heat into mechanical work...")
pipeline.train_reranker()

result = pipeline.query("How does a steam engine work?")
print(result.answer, result.grounding.overall_score, result.abstained)
```

### Swapping in a real LLM

```python
from verityrag import VerityRAGPipeline, LLMGenerator

def call_claude(prompt: str) -> str:
    # call your LLM provider of choice here
    ...

pipeline = VerityRAGPipeline(generator=LLMGenerator(complete_fn=call_claude))
```

The grounding checker runs on whatever the generator produces either way —
useful because a real LLM, unlike the extractive default, can genuinely
hallucinate even with correct evidence in context.

## License

MIT — see [LICENSE](LICENSE).
