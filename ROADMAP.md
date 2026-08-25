# Roadmap

Backlog for ongoing work, checked off as items land.

## Status
- [ ] **Retrieval sometimes fails to surface the right chunk — cause still
  unclear.** Re-ran against the current, tokenizer-fixed code rather than
  trust the stale 3-failure analysis — confirmed the failure set genuinely
  changed (now 5 failures matching 93.3% coverage, 3 retrieval misses + 2
  gate rejections; the Tesla/gender question moved from a retrieval miss
  to a gate rejection — the fix improved its lexical match enough to get
  retrieved, gate still says no). The 2 gate-rejected cases show clean
  separation from correct cases on `top1_dense`/`top3_mean_dense` (no
  range overlap at all) — but n=2 is too small to trust as a real,
  generalizable pattern, not the same confidence as the n=28/n=150
  samples used for similar findings earlier. Same conclusion as before,
  now on current data: too few failures at this coverage level for
  correlational analysis to say more.
- [ ] **Find what drives the sufficiency gate's remaining rejections.**
  Ruled out the lexical-score and lower-floor hypotheses, threshold
  tuning, calibration-set size (confirmed real, already near its
  plateau), and now a `top1_minus_top3mean_gap` feature too (see Done —
  had a real effect, unlike the floor experiment, but the effect looked
  net-harmful once actually inspected: it let through a confidently wrong,
  off-topic answer, not just a coverage number). No further concrete
  hypothesis identified — this has had more angles tried than any other
  thread this session, may be close to its practical ceiling.
- [ ] **Hybrid ranking: real imprecision, but the system already
  compensates for most of it.** Checked across all 150 answerable
  questions (not 1-2 anecdotes): raw `hybrid_retrieve()` puts the correct
  document at rank 1 only 61% of the time, but it's in the top-6 80% of
  the time — and actual coverage is 93.3%, because the reranker gets a
  second pass and the extractive generator can pull from any top-6
  candidate, not just rank 1. When the wrong document does win, dense
  score is involved 89% of the time vs. lexical 57% — the opposite of the
  one earlier anecdote (which used real embeddings, n=1). Given the
  system already shows real resilience to this, not clearly worth
  chasing further right now — the min-max-normalization-sensitivity
  theory is still untested if it becomes worth revisiting.
## Done
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