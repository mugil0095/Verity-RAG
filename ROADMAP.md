# Roadmap

A running backlog for ongoing work on this project, checked off as items
land. The point of this file is to make "update the project" a concrete,
pickable action instead of a vague standing intention — pick one open item,
do it, check it off, commit.

## Status
- [ ] **Re-tune the grounding checker for the new embedder** — `semantic_threshold=0.15`
  (grounding.py) was empirically chosen for HashingEmbedder's score
  distribution. avg_grounding_score dipped 1.0 → 0.967 with real
  embeddings even though the extractive generator is grounded by
  construction — a plausible sign this threshold isn't well-calibrated for
  the new embedding space. Worth its own measurement, not a guess.
- [ ] **Investigate the coverage regression with real embeddings** (96% → 88%)
  — best current explanation is real embeddings underweighting exact
  vocabulary overlap that SQuAD questions often have with their source
  sentence, but that's a hypothesis, not confirmed. Look at which of the 9
  wrongly-abstained questions differ from the 3 under HashingEmbedder.
- [ ] **Recover latency without losing stability** — `OMP_NUM_THREADS=1` /
  `MKL_NUM_THREADS=1` fixed a real crash but forces single-threaded
  execution everywhere, inflating the real-embeddings latency cost (p50
  ~1.56s) beyond what CPU transformer inference alone requires. Worth
  trying a real torch/numpy reinstall (a corrupted or mismatched install
  can cause this exact OpenMP conflict) to see if the underlying issue
  goes away rather than just staying worked around.
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
- [x] Real neural embeddings (`SentenceTransformerEmbedder`), measured
  end-to-end against the full corpus: hallucination guard 73.3% → **84%**
  (+10.7pp, confirms the core hypothesis), but coverage 96% → 88% (−8pp)
  and latency ~150ms → ~1,560ms p50 (~10x, partly a stability-workaround
  cost). Not flipped to the pipeline default — see README "Real embeddings,
  measured" for the full trade-off writeup.
- [x] Seeded both LightGBM models (`random_state=42`) — were unseeded,
  meaning every eval run had unquantified noise baked in on top of
  whatever an actual change (like the embedder swap) contributed. Verified
  two consecutive runs now produce byte-identical results.
- [x] Fixed two real Windows crashes getting real embeddings working:
  `STATUS_ACCESS_VIOLATION` from conflicting OpenMP runtimes (torch vs.
  MKL-linked scikit-learn/LightGBM) at model-load time, then a second one
  from runtime thread-pool contention under sustained interleaved use —
  see README "Design decisions".
- [x] 73 tests passing (+3 that need local internet to run)

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
- 2026-08-16 — Hit and fixed two real Windows crashes (OpenMP conflicts,
  load-time then runtime) getting `--real-embeddings` to actually complete.
  Fixed a progress-bar UX bug along the way (always-on spammed hundreds of
  near-instant progress bars during the query loop; now only shows for
  batches ≥20 items). Ran the full real-embeddings eval successfully:
  hallucination guard 73.3% → 84%, coverage 96% → 88%, latency ~10x.
  Seeded both LightGBM models afterward so this comparison (and all future
  ones) is reproducible rather than carrying unquantified run-to-run noise.
  Documented the full trade-off in README "Real embeddings, measured" —
  decided NOT to flip the pipeline default given the latency cost versus a
  system explicitly positioned as real-time.