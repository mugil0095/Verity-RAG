# Roadmap

Backlog for ongoing work, checked off as items land.

## Status
Nothing currently open — start a new thread below, or see Log for
context on anything past.

## Log

**2026-08-15** — Core pipeline built (chunking, hybrid retrieval,
LightGBM reranker, agentic loop, grounding, extractive generation,
streaming ingestion, calibrated sufficiency gate, FastAPI + Streamlit,
eval harness). Fixed a reformulation-drift bug (guard 9.3%→77.3%) and
O(n²) bulk ingestion (61s→2.7s).

**2026-08-16** — Added `SentenceTransformerEmbedder`. Fixed two Windows
OpenMP crashes. Real-embeddings eval: guard 73.3%→84%, coverage 96%→88%.

**2026-08-17** — Fixed two real memory bugs. Latency 150ms→57ms.

**2026-08-18** — Fixed a grounding-checker bug (0.967→1.0). Fixed CI
(wrong branch name); branch protection enabled.

**2026-08-19** — Diagnosed the coverage regression: sufficiency gate is
the bottleneck, not retrieval.

**2026-08-20** — Investigated 6 gate/retrieval hypotheses (most ruled
out). Built real LLM generation (Anthropic + Gemini) — fixed 3 real bugs
from live API calls (404/503/daily-quota). First real run: guard 100%,
coverage 50% (a genuine retrieval miss, not paraphrasing).

**2026-08-21** — Built the Elasticsearch swap with genuine incremental
indexing. Found + fixed a real tokenizer bug this way (~5pp guard
impact). Not yet faster than the default at this corpus's size (845
chunks).

**2026-08-22** — Re-verified numbers post-tokenizer-fix (real embeddings
guard 82.7%→96%). Tried + reverted a gate feature (real effect, but net-
harmful once inspected). Found a real ~15% real-embeddings latency
improvement (not adopted as the default — see README).

**2026-08-25** — Closed out the sufficiency-gate investigation for good
(54 failures, statistically significant per-feature but genuinely
probabilistic at the margin — not a missing signal). Ruled out the last
hybrid-ranking hypothesis. Load-tested concurrency for real (zero
errors). Deployed live to Streamlit Community Cloud, which surfaced and
fixed several real deployment-specific issues: shared pipeline singleton
for capacity, `usage_tracker.py` (GitHub-backed input logging), input
size limits, a reset-confirmation checkbox, a second memory fix on the
streaming-demo tab. Added a live extractive-vs-LLM comparison to the Ask
tab. Fixed a bare-`pytest` test-discovery bug (renamed
`concurrent_load_test.py` → `concurrent_load_check.py`,
`testpaths` added to `pyproject.toml`).

**2026-09-04** — Fixed two real CI failures neither reproducible without
care: a test-timing bug in the Reset test (`AppTest` checks a widget's
`disabled` state against the last completed run, not a pending one).
A second, LLM-comparison-test failure couldn't be reproduced locally
despite trying multiple `streamlit`/`google-genai` versions — made the
tests more diagnostic and pinned both dependencies as a hardening
measure, honestly not a confirmed fix for that specific failure.