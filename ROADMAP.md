# Roadmap

A running backlog for ongoing work on this project, checked off as items
land. The point of this file is to make "update the project" a concrete,
pickable action instead of a vague standing intention — pick one open item,
do it, check it off, commit.

## Status
- [ ] **Measure the real embeddings' actual impact** — `SentenceTransformerEmbedder`
  is done and confirmed working (see Done log), but not yet run against the
  full eval. Run `python -m verityrag.eval --real-embeddings` and update
  the README's headline numbers with whatever it reports — don't guess.
- [ ] **Real LLM generation** — wire `LLMGenerator` (generation.py) to an
  actual model via `complete_fn`. The interface and prompt construction are
  already built; this is the swap-in step. Once live, watch the grounding
  checker's `ungrounded_claims` output closely — a real LLM can hallucinate
  even with correct evidence in context, which the extractive default
  structurally can't.
- [ ] **Elasticsearch for the lexical index** — replace `rank_bm25`
  (indexing.py `LexicalIndex`) with real Elasticsearch/OpenSearch. Removes
  the full-rebuild-per-add cost on the streaming ingestion path (currently
  ~430ms/doc against a large index — see README "Design decisions").
- [ ] **Raise the hallucination-guard rate above 73.3%** — depends partly on
  the embedding swap above, but also worth its own pass: try richer
  sufficiency-gate features, or a larger/more diverse calibration set.
- [ ] **CI badge + branch protection** — once pushed to GitHub, add the
  Actions status badge to the README and require CI to pass before merge.

## Done
- [x] Core pipeline: chunking, hybrid retrieval, LightGBM reranker (ICT
  weak supervision), agent loop, grounding checker, extractive generation
- [x] Real-time streaming ingestion (thread-safe live index)
- [x] Calibrated sufficiency gate (replaced single-threshold heuristic)
- [x] Fixed reformulation-drift bug (hallucination guard 9.3% → 73.3%)
- [x] Fixed O(n²) bulk ingestion (61s → 2.7s)
- [x] FastAPI serving layer + root endpoint
- [x] Streamlit frontend + AppTest coverage
- [x] Full eval harness with disjoint calibration/test split
- [x] Real neural embeddings (`SentenceTransformerEmbedder`) — confirmed
  working on a real machine: paraphrase similarity 0.563 vs. unrelated
  -0.018. Deliberately NOT the pipeline default (would force every
  test/CI run to need internet + a model download); opt in explicitly via
  `run_eval(embedder=SentenceTransformerEmbedder())` or `--real-embeddings`
- [x] 70 tests passing (+3 that need local internet to run)

## Log
<!-- Add a dated one-line entry each time an item moves from Status to Done. -->
- 2026-08-15 — Roadmap created from README "Known limitations."
- 2026-08-16 — Added SentenceTransformerEmbedder + mocked wrapper tests.
  Confirmed working end-to-end on a real machine (paraphrase 0.563 vs.
  unrelated -0.018). Caught and fixed a real FutureWarning (renamed
  sentence-transformers method) and a real mocking bug (MagicMock
  auto-creates attributes, so hasattr-based fallback logic needs `del` to
  simulate an old API version in tests). Decided against flipping the
  pipeline default — added `--real-embeddings` to eval.py instead, so the
  upgrade is opt-in where it matters without slowing down every test run.