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
  (Note: this specific analysis predates the tokenizer fix below — the
  baseline is now 93.3%/5 failures, and the exact failure set has likely
  changed. The general conclusion, sample size is too small for this kind
  of analysis, almost certainly still holds, but the specific 3 questions
  named above may no longer be the current failures.)
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
  the time — and actual coverage is 93.3%, because the reranker gets a
  second pass and the extractive generator can pull from any top-6
  candidate, not just rank 1. When the wrong document does win, dense
  score is involved 89% of the time vs. lexical 57% — the opposite of the
  one earlier anecdote (which used real embeddings, n=1). Given the
  system already shows real resilience to this, not clearly worth
  chasing further right now — the min-max-normalization-sensitivity
  theory is still untested if it becomes worth revisiting.
- [ ] **Re-verify real-embeddings and real-LLM numbers against the fixed
  tokenizer.** The tokenizer bug fix (see Done) changed the default
  baseline (96%→93.3%, 73.3%→77.3%) — both `SentenceTransformerEmbedder`
  and the LLM-generator numbers were measured before this fix, using the
  same underlying `rank_bm25` lexical scoring, so they may shift slightly
  too. Not yet re-run.
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
  via a real health check). Real measurement against the live instance:
  see the two entries below.
- [x] **Measured the Elasticsearch swap against the real, running
  instance — two honest findings, neither the expected one.** (1) Eval
  numbers were NOT identical despite both being BM25-family scoring:
  guard 0.733 (rank_bm25) vs. 0.787 (Elasticsearch), coverage identical
  at 0.96. That gap is what surfaced a real tokenizer bug — see next
  entry. (2) At this corpus's actual size (845 chunks), Elasticsearch is
  not faster: `scripts/benchmark_lexical_index.py --elasticsearch`
  measured 206.0ms/doc (rank_bm25) vs. 221.7ms/doc (Elasticsearch) — a
  0.9x "speedup," i.e. slower. Elasticsearch's fixed per-call overhead
  (network round-trip + an explicit index refresh, required for
  real-time visibility) apparently exceeds rank_bm25's actual rebuild
  cost at this scale. The architectural principle (rank_bm25's cost
  grows with corpus size, Elasticsearch's doesn't) remains sound, but the
  crossover point where that pays off measurably hasn't been reached by
  this specific corpus — reported honestly rather than only benchmarked
  at a scale chosen to flatter the result.
- [x] **Found and fixed a real tokenizer bug via the Elasticsearch
  comparison above.** `rank_bm25`'s tokenizer was
  `text.lower().split()` — pure whitespace splitting, no punctuation
  stripping at all, so `"Tesla,"` and `"Tesla"` were different tokens and
  would never match each other. Confirmed directly
  (`_tokenize("Nikola Tesla, born in Smiljan...")` literally produced
  `"tesla,"` and `"smiljan,"` as tokens). Fixed to the same
  punctuation-stripping regex already used in `grounding.py`'s
  `_informative_tokens`, rather than inventing a different pattern.
  Closes ~74% of the guard-rate gap found above (0.733→0.773 with the
  fix, vs. Elasticsearch's 0.787) — strong, direct confirmation this was
  the real cause, not a coincidence. Real, honest trade-off: coverage
  dropped 96%→93.3% (3→5 wrongly abstained) — some of the old broken
  tokenizer's punctuation-attached matches were apparently helping a
  couple of answerable questions pass the sufficiency gate by accident.
  Kept the fix regardless, since correct tokenization isn't optional just
  because a bug happened to help sometimes. This changes the project's
  default baseline everywhere it's cited — updated README's headline
  numbers, comparison table, and known-limitations section; updated
  `eval_report.json`; updated ROADMAP's own current-facing references
  (not historical Log entries, which describe what was true at the time).
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
- [x] Fixed a reformulation-drift bug (hallucination guard 9.3% → 77.3%)
- [x] Fixed O(n²) bulk ingestion (61s → 2.7s)
- [x] FastAPI backend, Streamlit frontend, full eval harness
- [x] Real neural embeddings (`SentenceTransformerEmbedder`), measured
  against the baseline at the time: guard 73.3% → 84%, coverage 96% →
  88%, latency ~10x. Not the default — see README for the trade-off.
  Baseline since moved to 93.3%/77.3% (tokenizer fix, see below) — this
  specific comparison not yet re-run against the new baseline.
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
- 2026-08-21 — Built both remaining Elasticsearch measurement tools:
  `eval.py --elasticsearch` and `scripts/benchmark_lexical_index.py`.
  The benchmark's first run showed a real methodology problem before it
  showed a real result -- 2-3 outlier spikes (~2.5s) out of every 6-16
  samples were distorting the mean by ~5x. Ruled out garbage-collection
  timing directly (explicit `gc.collect()` before every call didn't
  change it) rather than assumed. Fixed by reporting median/p95 instead
  of mean, matching how every other latency number in this project is
  already reported. Clean rank_bm25 baseline: 225.1ms/doc median (n=16)
- 2026-08-21 -- Ran both Elasticsearch measurement tools against the real
  instance. Two honest findings, neither the expected one: eval numbers
  were NOT identical (guard 0.733 vs 0.787, despite both being BM25-family
  scoring), and Elasticsearch was SLOWER at this corpus's size (221.7ms/doc
  vs rank_bm25's 206.0ms/doc, a 0.9x "speedup"). The eval-number gap led
  to a real find: rank_bm25's tokenizer was text.lower().split(), no
  punctuation stripping, so "tesla," and "tesla" never matched. Fixed to
  the same regex pattern already used in grounding.py. Confirmed the fix
  was the real cause, not assumed: closes ~74% of the guard gap
  (0.733->0.773 vs Elasticsearch's 0.787). Honest trade-off: coverage
  96%->93.3% (some of the old bug's punctuation-attached matches were
  accidentally helping a couple of answerable questions pass) -- kept the
  fix anyway, since correct tokenization isn't optional. Updated every
  current-facing baseline reference in README/ROADMAP (not historical Log
  entries) and regenerated eval_report.json. The speed finding stands as
  reported, not chased into a fix -- Elasticsearch's fixed per-call
  overhead (network round-trip + required refresh) genuinely exceeds
  rank_bm25's rebuild cost at this specific corpus size; the architecture
  is still the right call for a corpus that grows much larger than this
  one, just not yet measurably faster at 845 chunks.