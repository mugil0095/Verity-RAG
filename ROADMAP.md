# Roadmap

Backlog for ongoing work, checked off as items land.

## Status
- [ ] **Retrieval sometimes fails to surface the right chunk — cause still
  unclear.** Traced the LLM-generator coverage gap to a genuine retrieval
  miss, not the paraphrase theory guessed earlier (ruled out — the LLM's
  rejected answers were honest "I don't know"s, not reworded correct
  ones; the gold chunk wasn't in the top-6 at all). Tested whether "broad
  topic, many chunks" explains it, using the full deterministic baseline
  rather than 2 anecdotal cases: doesn't hold up cleanly. Only 1 of the 3
  default-baseline failures (Nikola_Tesla, 92 docs) is from an
  above-median topic; Normans (45 docs) is actually below median. With
  only 3 failures total at 96% coverage, there isn't enough failure data
  for correlational analysis to find a real pattern here — a genuine
  sample-size limit, not a dead end to keep pushing on the same way.
- [ ] **Find what drives the sufficiency gate's remaining rejections.**
  Ruled out the lexical-score and lower-floor hypotheses, threshold
  tuning, and now calibration-set size too (see Done — confirmed real,
  but already near its plateau at current data levels). `top3_mean_dense`
  still looks like the real signal — a single strong match with weak
  supporting candidates gets under-trusted. Untried: reweighting that
  feature directly.
- [ ] **Hybrid ranking: real imprecision, but the system already
  compensates for most of it.** Checked across all 150 answerable
  questions (not 1-2 anecdotes): raw `hybrid_retrieve()` puts the correct
  document at rank 1 only 61% of the time, but it's in the top-6 80% of
  the time — and actual coverage is 96%, because the reranker gets a
  second pass and the extractive generator can pull from any top-6
  candidate, not just rank 1. When the wrong document does win, dense
  score is involved 89% of the time vs. lexical 57% — the opposite of the
  one earlier anecdote (which used real embeddings, n=1). Given the
  system already shows real resilience to this, not clearly worth
  chasing further right now — the min-max-normalization-sensitivity
  theory is still untested if it becomes worth revisiting.
- [ ] **Measure the Elasticsearch swap's actual numbers.** The wiring
  itself is now CONFIRMED against the real, running instance, not just
  mocks — ingested a document, queried it, got back the correct answer,
  not abstained, first real try. What's still open, same two questions
  as before: (1) does swapping it in change any eval numbers (shouldn't,
  BM25-family scoring either way, but "shouldn't" isn't "confirmed"), and
  (2) the actual incremental-indexing speedup vs. the documented
  ~430ms/doc rank_bm25 cost. One correct answer is real signal that the
  integration works, not the same claim as measured numbers.
- [ ] **Recover real-embeddings latency.** `OMP_NUM_THREADS=1` fixed a
  real Windows crash but forces single-threaded execution everywhere.
  Worth trying a clean torch/numpy reinstall to see if the underlying
  OpenMP conflict goes away instead of staying worked around.

## Done
- [x] **Elasticsearch for the lexical index.** Built `elasticsearch_index.py`
  (`ElasticsearchLexicalIndex`) — a drop-in swap for `rank_bm25`'s
  `LexicalIndex` with genuine incremental indexing, the actual point of
  this swap (rank_bm25 has no incremental API, forcing a full rebuild on
  every add). Required changing the shared interface from
  `rebuild(all_texts)` to `add_batch(new_ids, new_texts)` so the
  incremental benefit is real, not just a relabeled full rebuild — both
  implementations now expose the same interface, `LiveIndex`/
  `VerityRAGPipeline` both take an optional `lexical_index=` to swap in
  either one. Verified the current `elasticsearch-py` client API (9.5.0)
  by inspecting the installed package directly rather than assuming.
  12 new mocked tests. Set up Elasticsearch itself locally (native
  Windows install, security disabled, 1GB heap cap — confirmed working
  via a real health check). Real end-to-end measurement against the live
  instance is a separate, still-open item below.
- [x] **Calibration-set-size experiment: real effect, already near its
  plateau.** Built a proper learning curve (fixed, never-changing 30+30
  test set, increasing calibration size from 10→120) instead of a
  same-test-set-changes-each-time comparison. Real, clean result:
  coverage climbs sharply from 10→80 examples (+0.30 alone from 20→40),
  then completely flat from 80→120 (+0.000 both steps). Current
  production calibration size (75) lands almost exactly at that plateau —
  more calibration data is a genuine, confirmed lever, but there isn't
  much headroom left at the current 150-question data budget. Would need
  more labeled questions generated, not just a different split ratio, to
  test further.
- [x] Core pipeline: chunking, hybrid retrieval, LightGBM reranker,
  agentic loop, grounding checker, extractive generation
- [x] Real-time streaming ingestion, calibrated sufficiency gate
- [x] Fixed a reformulation-drift bug (hallucination guard 9.3% → 73.3%)
- [x] Fixed O(n²) bulk ingestion (61s → 2.7s)
- [x] FastAPI backend, Streamlit frontend, full eval harness
- [x] Real neural embeddings (`SentenceTransformerEmbedder`), measured:
  guard 73.3% → 84%, coverage 96% → 88%, latency ~10x. Not the default —
  see README for the trade-off.
- [x] Two Windows OpenMP crashes fixed getting real embeddings running
- [x] Two real memory bugs fixed (dead allocation, redundant matrix
  rebuild) — latency 150ms → 57ms as a side effect
- [x] Fixed a grounding-checker bug (chunk-level vs. sentence-level
  evidence matching) — confirmed against real embeddings: 0.967 → 1.0
- [x] CI fixed (was silently never running — wrong branch name) and
  branch protection enabled
- [x] Investigated the coverage regression properly instead of guessing:
  built a diagnostic script, found retrieval is fine and the sufficiency
  gate is the actual bottleneck. Tried a feature fix, verified it changed
  nothing, reverted it — see Status for the live thread.
- [x] Tested decision-threshold tuning — no improvement available, ruled
  out with real data
- [x] Fixed a test-suite memory bug on the dev machine (`gc.collect()` +
  a lighter fixture for one test)
- [x] Built real LLM generation: `llm_providers.py` with Anthropic and
  Gemini providers, wired into `VerityRAGPipeline`/`eval.py --real-llm`.
  Mocked tests for both; grounding check confirmed correct in both
  directions (catches a bad answer, passes a good one)
- [x] Fixed three real bugs found from live API calls, not tests: a
  stale model name (404), an uncaught server error (503), and an
  undetected daily quota limit (429) — each has its own regression test
  using the real exception type. Added `--max-test-questions` to make a
  small, honestly-labeled real measurement possible within free-tier
  limits
- [x] First complete real-LLM eval run (n=8+8, `gemini-3.6-flash`):
  guard 100%, coverage 50%, latency ~6.1s p50. Real numbers, small
  sample — coverage gap now an open item above.

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
- 2026-08-20 — Retracted the lexical-score hypothesis; `top3_mean_dense`
  is the real signal instead
- 2026-08-20 — Threshold tuning tested and ruled out
- 2026-08-20 — Fixed a test-suite memory bug
- 2026-08-20 — Built real LLM generation (Anthropic + Gemini)
- 2026-08-20 — Fixed three real bugs from live API calls (stale model
  404, uncaught 503, undetected daily-quota 429); added
  `--max-test-questions` for a feasible partial eval
- 2026-08-20 — First complete real-LLM run: guard 100%, coverage 50%
  on n=8+8 — coverage gap needs investigating
- 2026-08-20 — Added `AgentResult.raw_generated_text` (visible even when
  grounding rejects, unlike `answer`) and a diagnostic script to trace
  exactly why. Ran it for real: both rejected cases were the LLM
  correctly saying "I don't know" for evidence that was never actually
  retrieved (checked directly — the gold-answer chunk wasn't in the
  top-6). Paraphrase hypothesis retracted; this is a retrieval gap on
  broad, many-chunk topics, not an LLM or grounding-checker issue
- 2026-08-20 — Tested the "broad topic" theory against the full
  deterministic baseline instead of 2 anecdotal cases — doesn't hold up.
  Only 1 of 3 default-baseline failures is from an above-median-size
  topic. Genuine sample-size limit at 96% coverage (3 failures total),
  not a dead end worth pushing further the same way
- 2026-08-20 — Built a proper calibration-size learning curve (fixed
  test set, increasing calibration size 10→120) — real, clean result:
  coverage plateaus completely from 80 examples onward. Current
  production size (75) already sits almost exactly at that plateau
- 2026-08-20 — Checked hybrid ranking across all 150 answerable
  questions instead of 1-2 anecdotes: 61% exact rank-1 accuracy, but 80%
  in top-6, and dense score (not lexical) is behind the wrong-doc-wins
  cases 89% of the time — opposite of the one earlier anecdote. System
  already shows real resilience (96% coverage despite 61% rank-1), so
  not an urgent fix
- 2026-08-21 — Set up Elasticsearch locally (native Windows, security
  disabled, 1GB heap), confirmed working via a real health check. Built
  ElasticsearchLexicalIndex — required changing the shared lexical-index
  interface (rebuild(all_texts) → add_batch(new_ids, new_texts)) so the
  incremental-indexing benefit is genuine, not just a relabeled full
  rebuild. Verified the current elasticsearch-py client API directly
  before writing any code. 12 new mocked tests, 108 total passing. Real
  measurement against the live instance is a separate, still-open step
- 2026-08-21 — First real, live confirmation of the Elasticsearch
  integration: ingested a document into the real running instance,
  queried it, got the correct answer back, not abstained — first try,
  no errors. Wiring confirmed correct end-to-end; real eval-number and
  incremental-speedup measurements are still open, separate questions