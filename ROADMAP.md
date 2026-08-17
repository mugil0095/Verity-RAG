# Roadmap

A running backlog for ongoing work on this project, checked off as items
land. The point of this file is to make "update the project" a concrete,
pickable action instead of a vague standing intention — pick one open item,
do it, check it off, commit.

## Status
- [ ] **Investigate the coverage regression with real embeddings** (96% → 89.3%)
  — best current explanation is real embeddings underweighting exact
  vocabulary overlap that SQuAD questions often have with their source
  sentence, but that's a hypothesis, not confirmed. Look at which of the 8
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
- [x] Fixed two real memory bugs on a memory-constrained machine (found via
  actual `MemoryError` crashes, not review): a dead `np.vstack` line in
  `generate_ict_training_data` that allocated a 422MB array and never used
  it, and `hybrid_retrieve` rebuilding that same 422MB matrix from scratch
  on every single query instead of reusing the one `LiveIndex` already
  maintains. Fixing the second one dropped query latency **150ms → 57ms
  p50** — a real, measured win, not just a memory fix.
- [x] Fixed a real dilution bug in grounding.py: claims were matched
  against whole multi-sentence evidence chunks, not individual sentences.
  Measured directly (HashingEmbedder alone, no real encoder needed): a
  claim scored 1.0 against its own source sentence in isolation but only
  0.41 against a 5-sentence chunk containing that sentence plus four
  unrelated ones (2.4x dilution). Fixed by decomposing evidence into
  sentences before matching. `best_evidence_chunk_id` still correctly maps
  back to the parent chunk either way. **Confirmed against the real
  encoder: avg_grounding_score 0.967 → 1.0**, exactly matching
  HashingEmbedder's ceiling. Same fix moved coverage/guard by one question
  each in opposite directions (88%→89.3%, 84%→82.7%) — the agent's
  grounding-based final-abstention check was occasionally tripped by the
  dilution noise both ways, correctly for the wrong reason on one side.
- [x] 78 tests passing locally (75 in CI/offline — 3 need internet for the
  live model download, gracefully skipped there)

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
- 2026-08-17 — Found and fixed two real memory bugs via an actual full test
  suite run hitting `MemoryError` on a memory-constrained Windows machine
  (never surfaced in the build sandbox, which has more headroom): a dead
  `np.vstack` line in the reranker's ICT training that allocated a 422MB
  array and never used the result, and `hybrid_retrieve` reconstructing
  that same 422MB matrix from individual chunk vectors on every single
  query instead of reusing the one already maintained in `LiveIndex`.
  Added `LiveIndex.snapshot_with_matrix()` to expose it properly. Verified
  eval numbers unchanged (pure performance fix) — and query latency p50
  dropped from ~150ms to 57ms as a direct result.
- 2026-08-18 — Investigated the 1.0 → 0.967 avg_grounding_score gap seen
  with real embeddings instead of guessing at a new threshold number.
  Found and confirmed the actual cause empirically (using HashingEmbedder
  alone, no real encoder needed): matching a claim against a whole
  multi-sentence evidence chunk dilutes similarity to the one sentence it
  actually matches -- 2.4x dilution measured directly. Fixed by comparing
  claims against individual evidence sentences instead of whole chunks.
  2 new regression tests.
- 2026-08-18 — Confirmed the grounding fix against the real encoder:
  avg_grounding_score 0.967 -> 1.0, closing the gap completely. Coverage
  and hallucination guard each moved by one question in opposite
  directions (88%->89.3%, 84%->82.7%) -- explained, not just observed: the
  agent's grounding-based final-abstention check was occasionally tripped
  by the old dilution noise in both directions. Latency increased further
  (1557ms -> 1764ms p50), plausibly from the grounding check now embedding
  more/smaller evidence sentences instead of fewer/larger chunks per query.
  Updated README "Real embeddings, measured" with the full current numbers
  and explanation. 78 tests passing locally, all green.