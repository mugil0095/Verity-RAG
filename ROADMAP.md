# Roadmap

Backlog for ongoing work, checked off as items land.

## Status
Nothing currently open — start a new thread below, or see Log for
context on anything past.

## Log

**2026-08-15** — Roadmap created. Core pipeline built: chunking, hybrid
retrieval, LightGBM reranker, agentic loop, grounding checker, extractive
generation, real-time streaming ingestion, calibrated sufficiency gate,
FastAPI + Streamlit, eval harness. Fixed a reformulation-drift bug
(hallucination guard 9.3% → 77.3%) and an O(n²) bulk-ingestion bug
(61s → 2.7s).

**2026-08-16** — Added `SentenceTransformerEmbedder` (fixed a mocking bug
+ deprecation warning in its tests). Fixed two Windows OpenMP crashes
getting real embeddings running. Full real-embeddings eval: guard
73.3%→84%, coverage 96%→88%.

**2026-08-17** — Fixed two real memory bugs (dead allocation, redundant
matrix rebuild). Query latency 150ms → 57ms.

**2026-08-18** — Fixed a grounding-checker bug (chunk- vs. sentence-level
evidence matching), confirmed against real embeddings: 0.967 → 1.0. CI
fixed (was silently never running — wrong branch name); branch
protection enabled.

**2026-08-19** — Diagnosed the coverage regression: retrieval is fine,
the sufficiency gate is the bottleneck.

**2026-08-20** — Investigated 6 sufficiency-gate/retrieval hypotheses
(lexical score, threshold tuning, topic size, calibration size, hybrid
ranking); most ruled out with real data, calibration size confirmed real
but near-plateau. Built real LLM generation (`llm_providers.py`,
Anthropic + Gemini) — fixed 3 real bugs from live API calls, not tests
(stale model 404, uncaught server 503, undetected daily-quota 429).
First real run (n=8+8): guard 100%, coverage 50% — traced to a genuine
retrieval miss (LLM correctly said "I don't know"), not paraphrasing.

**2026-08-21** — Elasticsearch swap built (`elasticsearch_index.py`) with
genuine incremental indexing — required changing the shared interface,
not just adding a class alongside the old one. Confirmed working
end-to-end against a real running instance. Measuring it for real found
and fixed an actual tokenizer bug (`rank_bm25` never stripped
punctuation — worth ~5pp of guard rate) and showed Elasticsearch isn't
yet faster at this corpus's size (845 chunks: 206.0ms rank_bm25 vs.
221.7ms Elasticsearch) — architecture remains sound for a larger corpus,
just not the current one.

**2026-08-22** — Re-verified real-embeddings/real-LLM numbers against
the tokenizer fix: real embeddings, much bigger effect than the default
path (guard 82.7%→96%, explained by the classifier's known heavier
reliance on lexical score there); real LLM, unchanged, small-sample
noise only. Re-ran the retrieval-miss diagnostic on current (not stale)
data — same "sample too small to find a real pattern" conclusion.
Tried a `top1_minus_top3mean_gap` sufficiency-gate feature — real effect
this time (coverage 93.3%→94.7%, guard 77.3%→76.0%), but inspecting
*which* questions moved found it let through a confidently wrong,
off-topic answer invisible to the aggregate metric — reverted for being
net-harmful, not for having no effect. Found a real ~15% real-embeddings
latency improvement (`MKL_THREADING_LAYER=GNU` vs. `OMP_NUM_THREADS=1`),
confirmed across two full runs with identical accuracy — not adopted as
the default (the crash it works around needed *sustained* use to
manifest, and two runs is less evidence than the current fix's stable
track record); documented as an opt-in in the README instead.

**2026-08-25** — A long day; several separate, real threads closed out:
- Built an expanded, separate 400-question sample to get past the
  sample-size ceiling that had blocked every prior sufficiency-gate
  hypothesis. 54 real failures (10x more than any previous check); every
  gate feature statistically significant (Mann-Whitney p<0.01) but none
  cleanly separating rejected from correct — a real, conclusive answer:
  the problem is genuinely probabilistic at the margin, not a missing
  signal. Closes out this investigation thread.
- Tested the min-max-normalization-sensitivity theory directly (the last
  named hypothesis on the hybrid-ranking thread) instead of leaving it
  untested — checked whether the index-wide max score comes from outside
  the top-6 for all 29 wrong-rank-1 cases: only 3% (lexical), 0%
  (dense). Hypothesis doesn't hold; closes out this thread too.
- Load-tested concurrency for real instead of just reasoning about the
  locking code: real FastAPI server, genuine concurrent HTTP traffic,
  100 mixed ingest/query calls at once, zero errors, no lost updates.
- Deployed live to Streamlit Community Cloud. This surfaced several real,
  public-deployment-specific problems, fixed in sequence: (1) switched
  `app.py`'s pipeline to a shared singleton instead of per-session state
  (~1 concurrent session's worth of memory → several dozen on a
  ~1GB host) — caught and fixed a real test-isolation bug this
  introduced (`st.cache_resource` is scoped to the process, not to
  individual `AppTest` instances). (2) Built `usage_tracker.py` — logs
  real demo inputs to a separate, private GitHub repo, fails silently on
  any error so it never blocks a visitor's actual request; later fixed
  it again to run the GitHub commit on a background thread instead of
  blocking every submission on a synchronous call with a 10s timeout.
  (3) Added input size limits to both `app.py` and `api.py`, and a
  confirmation checkbox before the shared "reset" button fires — a
  shared pipeline means one large submission or one careless click now
  affects every current visitor, not just one session. (4) Found a
  second, separate memory gap: the streaming-demo tab's per-visitor
  index was ingesting 528 docs per visitor (nearly as large as the
  shared main corpus) — reduced to a 9-topic, 372-doc subset, verified
  the exact scripted demo scenario still works correctly both directly
  and through the real `AppTest` flow before trusting it.
- Added a live, side-by-side extractive-vs-LLM comparison to the Ask
  tab — same retrieved evidence, both generators, no duplicated memory
  (shares the main pipeline's index/embedder/reranker/gate). Caught two
  real bugs before shipping: `st.secrets.get()` raises
  `StreamlitSecretNotFoundError` rather than returning `None` with no
  `secrets.toml` configured (would have crashed the app), and a
  `str_replace` edit left an orphaned code block behind during a
  refactor (caught by a syntax check before running anything).
- Fixed a real, user-reported bug: bare `pytest` (no arguments)
  accidentally collected `scripts/concurrent_load_test.py` as isolated
  test functions (filename + function names both matched pytest's
  discovery conventions), never starting the real server it needs.
  Fixed with `testpaths = ["tests"]` in `pyproject.toml` plus renaming
  the file to `concurrent_load_check.py`.

**2026-09-04** — Fixed a real CI failure (`test_reset_button_clears_the_index`)
that hadn't shown up locally: `at.sidebar.checkbox[0].set_value(True)` was
called with no `.run()` after it, then the very next line grabbed and
clicked the Reset button — but `AppTest` checks a button's `disabled`
state against the *last completed run*, not any pending change, so the
button still reflected its pre-checkbox-change (`disabled=True`) state
and the click correctly errored. Fixed by chaining `.run(timeout=30)`
right after the checkbox's `set_value()`, matching the pattern already
used correctly elsewhere in the same file. Notably didn't reproduce with
an older locally-cached `streamlit` (1.61.1) but did with a newer one
(1.62.0, likely closer to what CI installs) — `requirements.txt` doesn't
pin `streamlit`'s version, so this was a real latent bug in the test
itself (relying on an older version's more lenient behavior), not
something introduced by an unrelated change. Verified with the newer
version installed: all 122 tests pass.