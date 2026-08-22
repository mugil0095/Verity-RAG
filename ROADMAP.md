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
- [ ] **Recover real-embeddings latency.** `OMP_NUM_THREADS=1` fixed a
  real Windows crash but forces single-threaded execution everywhere.
  Worth trying a clean torch/numpy reinstall to see if the underlying
  OpenMP conflict goes away instead of staying worked around.

## Done
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