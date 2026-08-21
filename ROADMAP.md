# Roadmap

Backlog for ongoing work, checked off as items land.

## Status
- [ ] **Retrieval fails to surface the right chunk among many
  topically-similar ones for broad topics.** Traced the LLM-generator
  coverage gap to its actual root, not the paraphrase theory guessed
  earlier (that's now ruled out — checked directly, the LLM's rejected
  answers were honest "I don't know"s, not reworded correct ones). Both
  grounding-rejected cases from a real run: the gold-answer chunk simply
  wasn't in the top-6 retrieved candidates at all, for questions about
  Super Bowl 50 (a broad topic spanning many separate chunks). Same
  failure would hit the extractive generator too — it just can't say "I
  don't know," so it silently produces a wrong answer instead of an
  honest abstention. Connects directly to the two items below rather than
  being a separate LLM-specific issue.
- [ ] **Find what drives the sufficiency gate's remaining rejections.**
  Ruled out the lexical-score and lower-floor hypotheses, and threshold
  tuning (no cutoff beats the current default on both coverage and
  guard). `top3_mean_dense` looks like the real signal — a single strong
  match with weak supporting candidates gets under-trusted. Untried:
  reweighting that feature, or a larger calibration set.
- [ ] **Hybrid ranking occasionally promotes the wrong document to rank 1**
  — a separate issue from the gate above, worth its own look at the
  `lexical_weight`/`dense_weight` balance in `retrieval.py`.
- [ ] **Recover real-embeddings latency.** `OMP_NUM_THREADS=1` fixed a
  real Windows crash but forces single-threaded execution everywhere.
  Worth trying a clean torch/numpy reinstall to see if the underlying
  OpenMP conflict goes away instead of staying worked around.
- [ ] **Elasticsearch for the lexical index** — replace `rank_bm25` to
  remove the full-rebuild-per-add cost on the streaming path.

## Done
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