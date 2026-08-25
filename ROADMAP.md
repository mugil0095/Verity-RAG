# Roadmap

Backlog for ongoing work, checked off as items land.

## Status
Nothing currently open — see Log for what's next up for grabs, or start a
new thread.

## Done
- [x] **Made app.py's pipeline a shared singleton instead of per-session
  state, for real hosted-deployment capacity.** Each browser session
  previously got its own independent `VerityRAGPipeline()` — a nicer
  experience (one visitor's actions never affect another's), but each
  fully-loaded session duplicates the ~422MB vector matrix (845 chunks x
  65536-dim `HashingEmbedder` vectors, confirmed directly). On a
  memory-constrained free host (~1GB), that's the difference between
  supporting roughly 1 concurrent session and several dozen. Switched to
  `@st.cache_resource` (one instance for every visitor, matching how
  `api.py` already works) — the UI now says explicitly that loading,
  resetting, and adding documents affect the shared state for everyone,
  rather than changing this silently. The separate streaming-demo
  pipeline stays per-session on purpose (it needs each visitor to see a
  fresh "before streaming" state for the demo to mean anything).
  Found and fixed a real test-isolation bug this introduced: two separate
  `AppTest` instances in the same pytest process got the literal same
  cached pipeline object (confirmed directly via `id()`), so tests were
  only passing because of file ordering, not genuine isolation — fixed
  with an explicit `st.cache_resource.clear()` between tests, scoped to
  just this test file.
- [x] **Actually load-tested concurrency instead of just reasoning about
  the locking code.** `LiveIndex` uses `threading.RLock()` around both
  reads and writes — designed for concurrent ingestion/querying, but
  never verified under real load. Built
  `scripts/concurrent_load_test.py`: starts the real FastAPI server
  (uvicorn subprocess) and fires genuine concurrent HTTP traffic at it,
  not just calling Python objects directly. 30 simultaneous `/ingest`
  calls (final index size exactly matched what was sent — no lost
  updates), 30 simultaneous `/query` calls, 100 concurrent mixed
  ingest+query calls — zero errors across all of it, at two different
  scales. The design holds up: `VectorIndex.add()` uses `np.vstack` to
  build a new array rather than mutate one in place, so a reader's
  snapshot stays valid even through a concurrent write.
- [x] **Ruled out the last named hypothesis on the hybrid-ranking
  thread: min-max-normalization sensitivity.** The theory was that
  normalizing lexical/dense scores over the ENTIRE index (not just the
  top-k) could let some unrelated, high-scoring chunk elsewhere in the
  corpus distort the scale for a given query, favoring the wrong document.
  Tested directly across all 150 answerable questions: for the 29 cases
  where the wrong document wins rank 1, checked whether the index-wide
  max lexical/dense score came from within the actual top-6 candidates or
  from somewhere else entirely. Lexical: 1/29 (3%) from outside. Dense:
  0/29 (0%). The hypothesis doesn't hold — in virtually every case, the
  top-6 candidates already contain the highest-scoring chunks for that
  query (unsurprising in retrospect: that's how they became top-6), so
  the normalization scale isn't being distorted by some far-away outlier.
  The "wrong document wins" pattern is a genuine, head-to-head competition
  among the real candidates, decided by the actual 0.4/0.6
  lexical/dense weighting — not a normalization artifact. Closes out the
  hybrid-ranking investigation thread with no remaining open questions.
- [x] **Closed out the sufficiency-gate/retrieval-miss investigation with
  a genuinely conclusive answer, not another inconclusive attempt.** Every
  prior hypothesis on this thread was blocked by the same wall: only 5
  failures at 93.3% coverage is too few to trust. Built
  `scripts/expanded_coverage_sample.py` — a separate, additive sample
  (400 questions, same 12 topics, higher per-paragraph cap than
  `build_corpus.py`'s default of 2; doesn't touch the existing
  150-question baseline any other measured number depends on) — to
  actually get past the sample-size ceiling instead of accepting it.
  Result: 54 real failures (10x the previous sample), and a proper
  Mann-Whitney significance test on all 5 gate features against 346
  correct cases. Every single feature is genuinely, statistically
  significant (p<0.01, most below p<0.000001) — confirming these
  features do carry real signal, not coincidence — but every one still
  has real range overlap between the groups. Honest, well-evidenced
  conclusion: the gate isn't missing an obvious signal it should be
  using — the underlying problem is genuinely, irreducibly probabilistic
  at the decision boundary, which is exactly why no single feature
  reweighting or threshold adjustment ever produced a clean win across
  this whole investigation. This is the actual answer, not a dead end —
  it explains the previous 6 attempts' consistent lack of a clean fix.
- [x] **Found a real, reproducible ~15% latency improvement for real
  embeddings — not yet adopted as the default.** `MKL_THREADING_LAYER=GNU`
  (instead of the current `OMP_NUM_THREADS=1`) measured p50 2175ms→1855ms,
  p95 4087ms→3443ms, confirmed across two separate full-eval runs with
  identical coverage/guard (0.92/0.96) both times — a real speedup, not
  noise, with zero accuracy cost. Kept `OMP_NUM_THREADS=1` as the actual
  default anyway: the crash it was fixing specifically needed *sustained*
  use to manifest (~100 questions into a run), and two successful
  ~150-question runs, while a reasonably good sign, is meaningfully less
  runtime than `OMP_NUM_THREADS=1` has accumulated stably across this
  entire session. A crash is a worse outcome than "a bit slow," so this
  needs more sustained testing (e.g. the Streamlit streaming demo running
  for an extended period) before it's worth the risk of switching the
  default. Documented as an available, faster opt-in instead — see README.
- [x] **Tested `top1_minus_top3mean_gap` for the sufficiency gate —
  reverted, real effect but net-harmful.** Added as an additional feature
  (top1_dense minus top3_mean_dense), meant to capture "isolated strong
  match" cases without removing any existing signal — lower risk than the
  earlier floor experiment. Had a genuine, measured effect on the trained
  classifier (unlike that floor experiment, which had none): coverage
  93.3%→94.7%, guard 77.3%→76.0%. But checking which SPECIFIC questions
  moved — not just the aggregate counts — showed several individual
  predictions shifting in both directions, not a clean net +1/-1. One
  newly-"answered" case was a real, confirmed problem: a confidently wrong,
  completely off-topic answer (about Luther's biographer, for a question
  about where Luther focused his reform efforts) counted as a coverage
  success purely because the extractive generator pulled verbatim text
  from *some* document. `coverage_rate` doesn't check answer correctness,
  so this failure mode was invisible to the headline metric even though
  it's exactly what this project exists to prevent. Reverted for a
  different reason than the floor experiment: not because it did nothing,
  but because its real effect looked harmful once actually inspected.
- [x] **Re-verified real-embeddings and real-LLM numbers against the
  fixed tokenizer.** Real embeddings: much bigger effect than the
  default path (guard 82.7%→96%, coverage 89.3%→92%, keyword-hit
  70.1%→76.8% — all improved together, no trade-off this time). Not a
  coincidence: the real-embeddings-trained sufficiency classifier relies
  on lexical score as its single most important feature (confirmed
  earlier via `feature_importances_`), more than the `HashingEmbedder`
  classifier does — a bug in exactly that layer naturally hit harder
  where it mattered most. Real LLM (n=8+8): coverage/guard exactly
  unchanged (0.5, 1.0); the one number that moved (grounding 1.0→0.875)
  is explained by simple arithmetic on 4 attempted questions, within
  already-expected LLM run-to-run noise.
- [x] Core pipeline: chunking, hybrid retrieval, LightGBM reranker,
  agentic loop, grounding checker, extractive generation
- [x] Real-time streaming ingestion, calibrated sufficiency gate
- [x] Fixed a reformulation-drift bug (hallucination guard 9.3% → 77.3%)
- [x] Fixed O(n²) bulk ingestion (61s → 2.7s)
- [x] FastAPI backend, Streamlit frontend, full eval harness
- [x] Real neural embeddings (`SentenceTransformerEmbedder`) — see
  README for the current numbers (re-verified after the tokenizer fix,
  entry above). Not the default; latency is the real trade-off.
- [x] Two Windows OpenMP crashes fixed getting real embeddings running
- [x] Two real memory bugs fixed (dead allocation, redundant matrix
  rebuild) — latency 150ms → 57ms
- [x] Fixed a grounding-checker bug (chunk- vs. sentence-level evidence
  matching) — confirmed against real embeddings: 0.967 → 1.0
- [x] CI fixed (was silently never running — wrong branch name); branch
  protection enabled
- [x] Investigated the coverage regression across 6 hypotheses (feature
  engineering, lexical score, threshold tuning, topic size,
  calibration-set size, hybrid ranking). Most ruled out with real data;
  calibration size confirmed real but already near its plateau (75
  production size ≈ where the curve goes flat); hybrid ranking shows real
  imprecision (61% rank-1 accuracy) the reranker/generator already absorb
  most of (96%→93.3% actual coverage regardless).
- [x] Fixed a test-suite memory bug (`gc.collect()` + a lighter fixture)
- [x] Built real LLM generation (`llm_providers.py`, Anthropic + Gemini).
  Fixed 3 real bugs found from live API calls, not tests: a stale model
  name (404), an uncaught server error (503), an undetected daily quota
  (429) — each with its own regression test using the real exception
  type. Added `--max-test-questions` for feasible partial evals. First
  real run (n=8+8): guard 100%, coverage 50% — traced to a genuine
  retrieval miss (the LLM correctly said "I don't know"), not the
  paraphrase theory guessed initially.
- [x] Elasticsearch swap: built `elasticsearch_index.py` with genuine
  incremental indexing (required changing the shared interface, not just
  adding a class alongside the old one). Set up Elasticsearch locally,
  confirmed working end-to-end against the real instance. Measuring it
  for real found and fixed an actual tokenizer bug (`rank_bm25` never
  stripped punctuation — worth ~5pp of guard rate) and showed
  Elasticsearch isn't yet faster at this corpus's size (845 chunks) —
  fixed per-call overhead exceeds `rank_bm25`'s rebuild cost here; the
  architecture remains sound for a corpus that grows much larger.

## Log
- 2026-08-15 — Roadmap created
- 2026-08-16 — Added `SentenceTransformerEmbedder`; fixed a mocking bug
  and a deprecation warning in its tests
- 2026-08-16 — Fixed two Windows OpenMP crashes; full real-embeddings
  eval run (guard 73.3%→84%, coverage 96%→88%)
- 2026-08-17 — Fixed two memory bugs; query latency 150ms→57ms
- 2026-08-18 — Fixed grounding dilution bug; confirmed against real
  embeddings (0.967→1.0)
- 2026-08-18 — Fixed CI (wrong branch name, matrix→single Python
  version); enabled branch protection
- 2026-08-19 — Diagnosed the coverage regression: retrieval is fine, the
  sufficiency gate is the bottleneck
- 2026-08-20 — Investigated 6 sufficiency-gate/retrieval hypotheses
  (lexical score, threshold tuning, topic size, calibration size, hybrid
  ranking); most ruled out with real data, calibration size confirmed
  real but near-plateau
- 2026-08-20 — Built real LLM generation (Anthropic + Gemini); fixed 3
  real bugs from live API calls (404/503/daily-quota); first real run
  (guard 100%, coverage 50%) traced to a genuine retrieval miss, not
  paraphrasing
- 2026-08-21 — Set up Elasticsearch locally; built
  `ElasticsearchLexicalIndex` + measurement tools; confirmed working
  end-to-end
- 2026-08-21 — Measured Elasticsearch for real: found and fixed a real
  tokenizer bug (~5pp guard impact); found it's not yet faster at this
  corpus's size
- 2026-08-22 — Re-verified real-embeddings/real-LLM numbers against the
  tokenizer fix. Real embeddings: much bigger effect than the default
  path (guard 82.7%→96%), explained by the classifier's known heavier
  reliance on lexical score there. Real LLM: unchanged, small-sample
  noise only
- 2026-08-22 — Re-ran the retrieval-miss diagnostic against the current
  tokenizer-fixed code (was stale, predated the fix). Failure set
  genuinely changed (now 5, not 3) — same "sample size too small to
  find a real pattern" conclusion as before, just confirmed on current
  data instead of stale data
- 2026-08-22 — Tried a `top1_minus_top3mean_gap` feature for the
  sufficiency gate; real effect this time (unlike the floor experiment),
  but inspecting which specific questions moved found it let through a
  confidently wrong, off-topic answer — reverted for being net-harmful,
  not for having no effect
- 2026-08-22 — Found a real ~15% real-embeddings latency improvement
  (`MKL_THREADING_LAYER=GNU` vs. the current `OMP_NUM_THREADS=1`),
  confirmed across two full-eval runs with identical accuracy. Not
  adopted as the default yet — the crash it works around needed
  sustained use to manifest, and two runs is less evidence than
  `OMP_NUM_THREADS=1`'s stable track record across this whole session.
  Documented as an opt-in instead of risking a regression for 15%
- 2026-08-25 — Built an expanded, separate 400-question sample (same
  topics, higher per-paragraph cap) to get past the sample-size ceiling
  that blocked every prior sufficiency-gate hypothesis. 54 real failures
  (10x more than any previous check), every gate feature statistically
  significant (Mann-Whitney p<0.01) but none cleanly separating the
  groups — a real, conclusive answer: the problem is genuinely
  probabilistic at the margin, not a missing signal
- 2026-08-25 — Tested the min-max-normalization-sensitivity theory
  directly (last named hypothesis on the hybrid-ranking thread) instead
  of leaving it as an untested idea. Checked whether the index-wide max
  score comes from outside the top-6 for all 29 wrong-rank-1 cases —
  only 3% (lexical) and 0% (dense) did. Hypothesis doesn't hold; closes
  out the hybrid-ranking thread with no open questions remaining
- 2026-08-25 — Load-tested concurrency for real instead of just
  reasoning about the locking code: real FastAPI server, genuine
  concurrent HTTP traffic, 100 mixed ingest/query calls at once, zero
  errors, no lost index updates
- 2026-08-25 — Switched app.py to a shared pipeline singleton for hosted
  deployment capacity (~1 concurrent session -> several dozen on a
  memory-constrained host). Found and fixed a real test-isolation bug
  this caused (two separate AppTest instances shared the same cached
  object) — tests were passing due to file ordering, not real isolation