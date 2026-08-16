# Roadmap

A running backlog for ongoing work on this project, checked off as items
land. The point of this file is to make "update the project" a concrete,
pickable action instead of a vague standing intention — pick one open item,
do it, check it off, commit.

## Status
- [ ] **Real neural embeddings** — swap `HashingEmbedder` (embedding.py) for
  a real dense encoder (sentence-transformers, or an embeddings API) behind
  the same `.embed()` interface. Highest-leverage single change: it's the
  documented ceiling on the 73.3% hallucination-guard rate and the 68%
  keyword-hit rate (see README "Known limitations").
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
- [x] 66 tests passing

## Log
<!-- Add a dated one-line entry each time an item moves from Status to Done. -->
- 2026-08-15 — Roadmap created from README "Known limitations."
