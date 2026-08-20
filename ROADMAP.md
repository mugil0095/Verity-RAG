# Roadmap

A running backlog for ongoing work on this project, checked off as items
land. The point of this file is to make "update the project" a concrete,
pickable action instead of a vague standing intention — pick one open item,
do it, check it off, commit.

## Status
- [ ] **`top3_mean_dense` is the real signal — lexical hypothesis retracted.**
  Ran a proper comparison instead of trusting `feature_importances_` alone:
  captured feature vectors for all 67 correctly-answered questions too
  (not just the 7 rejections), and compared actual ranges. `top1_lexical_raw`
  (the feature flagged last session as most important, 73) turned out NOT
  to separate the groups — rejected median (17.46) was actually *higher*
  than correct median (16.06), 69% range overlap. LightGBM's default
  feature importance mostly reflects split *count*, and an unbounded raw
  BM25 score naturally gets split on more than a bounded [0,1] one — high
  importance isn't the same as actually separating classes, worth
  remembering next time this method gets used. `top3_mean_dense` shows the
  real separation instead: only 18% range overlap, rejected mean 0.33 vs.
  correct mean 0.57 — consistent with the original Raouliii finding (one
  excellent top1 match, weak rank-2/3 support dragging the average down).
  `n_candidates_above_floor` reconfirmed dead beyond doubt: identical
  constant `[6.0, 6.0]` range for BOTH groups now, not just the rejections.
  Decision-threshold tuning also tested and ruled out (see Done log) — the
  only two untried levers left are a larger calibration set, or a feature
  that more directly captures "isolated strong match vs. broad support"
  than `top3_mean_dense` alone.
- [ ] **Separately: hybrid ranking sometimes promotes the wrong document**
  — 2 of the 8 wrongly-abstained cases didn't have the correct document at
  rank 1 at all. In one (Tesla gender question), the wrong rank-1 document
  had a WEAKER raw semantic score (0.2385) than the correct rank-2 document
  (0.4756), meaning it won only on lexical score. This is a hybrid-weight
  or reranker problem, not a gate-calibration one — a different fix,
  independent of the item above.
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

## Done
- [x] **Decision-threshold tuning: tested, no improvement available.** Built
  `scripts/sweep_threshold.py` to check whether a different cutoff on the
  already-trained classifier would help, without retraining. Result: a
  clean, monotonic trade-off curve — checked explicitly, no threshold in
  [0.10, 0.50] beats the current default (0.5) on BOTH coverage and guard
  simultaneously. Confirmed the hop-0-only approximation's known gap
  against real full-pipeline numbers is a stable, expected pattern, not
  noise: reformulation adds ~+0.05 coverage and costs ~-0.06 guard at the
  same threshold, consistent in direction and rough magnitude across BOTH
  HashingEmbedder (approx 0.907/0.773 vs real 0.96/0.733) and real
  embeddings (approx 0.840/0.893 vs real 0.893/0.827). This is the third
  hypothesis ruled out by real testing in this investigation
  (`n_candidates_above_floor`, lexical score, now threshold) — the pattern
  across all three: the classifier isn't obviously miscalibrated, it's
  making a reasonable trade-off given its current features/data. The
  constraint is information available to it, not a bug in how it's used.
- [x] Fixed a real MemoryError on the actual machine that's been running
  these evals, not just a theoretical concern: running the full test suite
  in one pytest process hit crashes from HashingEmbedder's dense 65536-dim
  vectors accumulating across several heavy tests (`test_app.py` alone
  does ~5 separate full-corpus loads). Reducing embedding dimensionality
  and shrinking the regression-test corpus were both tested directly and
  rejected — the first measurably changes eval numbers (96%→92% coverage
  at 16384 features), the second nearly destroys the reformulation-drift
  bug's detectability (0.200 vs 0.050, versus 0.4 vs 0.0 at the current
  size). Added `tests/conftest.py` (`gc.collect()` after every test) —
  fixed 2 of 3 original failures, confirmed by real before/after test
  output, not assumed. The third (`test_reset_button_clears_the_index`,
  the single most cumulative test, last in its file) needed its own fix:
  swapped the full-corpus button for the already-tested lightweight
  manual-document form, since Reset's correctness never depended on how
  much was in the index. 78 tests passing on the real machine now.
- [x] Root-caused and reverted the `n_candidates_above_floor` relative-floor
  fix — confirmed, twice over, that this feature is dead for real
  embeddings and safe to leave alone. `scripts/diagnose_coverage.py`
  categorized the 8 wrongly-abstained real-embeddings questions instead of
  guessing: 7 of 8 were gate rejections with the correct document already
  retrieved, often at rank 1 with a strong score — overturning the
  original "real embeddings underweight vocabulary overlap" hypothesis.
  Tried a relative floor for `n_candidates_above_floor` (constant at 6.0
  for every case with the old absolute floor) — empirically confirmed it
  changed nothing: the classifier's output probability was IDENTICAL
  before and after for every rejected case (Raouliii's feature value moved
  6.0→2.0, its probability stayed exactly 0.1448). Reverted cleanly,
  restoring the original 96%/73.3% baseline exactly. Independently
  reconfirmed via real `feature_importances_`: exactly **0** on real
  embeddings, weakest of 5 on HashingEmbedder too.
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
- [x] Fixed the CI workflow itself: it targeted branch `main`, but this
  repo's actual default branch is `master` — CI had never once run,
  silently, since the workflow was written. Also split into a required
  fast/deterministic job and a separate non-blocking job for the real
  HuggingFace model download (rate-limit-prone on shared CI runners,
  flaky for reasons unrelated to code correctness), and added workflow
  timeouts so a stuck job fails fast instead of running for hours.
  Simplified away an accidental matrix (2 Python versions) that produced
  ambiguous check names (`test (3.11)` / `test (3.12)`, not `test`) and
  blocked branch protection from finding the check at all.
- [x] Enabled branch protection on `master`, requiring the `test` job to
  pass before merge — confirmed against a real successful run
  (`https://github.com/mugil0095/Verity-RAG/actions/runs/32063731699`),
  not just configured and assumed working. Added the CI status badge to
  the top of README.md.

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
- 2026-08-18 — Fixed the CI workflow: it targeted branch `main`, but the
  repo's actual default branch is `master`, so CI had silently never run
  at all since it was written, long before Streamlit or real embeddings
  existed in this project. Rewrote it for the current state: split into a
  required fast/deterministic job (excludes the live HuggingFace-download
  test, which is rate-limit-prone and flaky on shared CI runners) and a
  separate non-blocking job for that live test, added the OpenMP env vars
  defensively, added workflow timeouts. Validated by parsing the YAML
  directly, not just visual review.
- 2026-08-18 — Simplified CI's Python matrix down to a single version
  after it caused a real, confusing problem: matrix jobs report as
  `test (3.11)` / `test (3.12)`, not `test`, so branch protection's status
  check search couldn't find a plain `test` to require. Verified the fix
  by fetching the actual latest workflow run directly rather than trusting
  it worked — confirmed exactly one job named `test`, status Success.
  Branch protection enabled on `master` requiring it. CI badge added to
  README. This closes out the CI roadmap item completely: a real,
  currently-running, merge-blocking check, not just a committed YAML file.
- 2026-08-19 — Built scripts/diagnose_coverage.py to investigate the
  coverage regression properly instead of guessing. Overturned the
  original hypothesis: 7 of 8 wrongly-abstained real-embeddings questions
  had the correct document already retrieved (5 at rank 1, with dense
  scores from 0.20-0.67 -- a confidence-calibration problem, not a
  retrieval one. Confirmed one concrete cause: n_candidates_above_floor
  used a fixed absolute floor (0.08) that was measurably CONSTANT (6.0,
  every single case) with real embeddings -- a dead feature. Fixed to a
  relative floor, empirically tuned (0.4, chosen to preserve guard rate)
  since no fraction tested fully recovered the HashingEmbedder baseline.
  Real cost, not yet a confirmed win: coverage 96%->90.7%, guard
  73.3%->70.7% on the DEFAULT path. Deliberately not updating README/other
  ROADMAP numbers yet -- this trade-off needs real-embeddings confirmation
  first (does it actually fix the 5 gate-rejected questions?) before
  deciding whether to keep it. Also found, separately: 2 of 8 cases had
  the WRONG document outranking the correct one at rank 1 (hybrid-ranking
  problem, not gate-calibration) -- needs its own investigation.
- 2026-08-19 — Reverted the relative-floor fix after confirming it doesn't
  work: ran scripts/diagnose_coverage.py --real-embeddings again and
  compared probabilities directly. n_candidates_above_floor's value did
  genuinely change for the Raouliii case (6.0 -> 2.0, proving the fix
  itself was implemented correctly), but the trained classifier's output
  probability for that exact case was IDENTICAL before and after (0.1448),
  as were all 6 other previously-rejected cases (identical to 4 decimal
  places -- checked programmatically, not eyeballed). The classifier
  places ~zero weight on this feature regardless of how it's computed, so
  the fix bought a confirmed regression (coverage 96%->90.7%, guard
  73.3%->70.7%) for zero confirmed benefit. Reverted cleanly -- confirmed
  the original 96%/73.3% baseline is restored exactly, and removed the 2
  tests that were locking in the now-abandoned design. Added real
  LGBMClassifier.feature_importances_ output to the diagnostic script
  instead of continuing to infer the driver from correlations -- next
  real-embeddings run will show directly which feature the trained
  classifier actually relies on.
- 2026-08-19 — Got the real feature_importances_ instead of continuing to
  guess: on the real-embeddings-trained classifier, top1_lexical_raw is
  the single most relied-on feature (73), ahead of top3_mean_dense (67),
  top1_dense (52), dense_gap_top1_top2 (48) -- and n_candidates_above_floor
  scores exactly 0, independently confirming yesterday's revert was
  correct. Notably different from HashingEmbedder, where dense_gap led
  (71) and lexical was third (51) -- lexical's relative importance roughly
  doubled with real embeddings despite BM25 scoring being completely
  embedder-independent, suggesting the classifier compensates for noisier
  dense features by leaning on the one signal that didn't get noisier.
  Plausible, not independently confirmed. Verified the rejected cases show
  a real but imperfect lexical-score/probability correlation rather than
  assuming one. Stopping here for today rather than rushing a fix on top
  of an already-long investigation -- next attempt (reweighting features
  or a larger calibration set) gets the same build-it-then-verify-with-
  real-embeddings treatment the last one did before being trusted.
- 2026-08-20 — Retracted the lexical hypothesis from the entry above.
  Enhanced scripts/diagnose_coverage.py to capture feature vectors for all
  67 correctly-answered questions, not just the 7 rejections -- without a
  real baseline, "lexical looks low" was never actually tested against
  anything. Real comparison: top1_lexical_raw has 69% range overlap
  between rejected and correct groups, and the rejected median (17.46) is
  actually HIGHER than the correct median (16.06) -- the opposite of the
  hypothesis. Root cause of the bad inference: LightGBM's default feature
  importance mostly reflects split count, and an unbounded raw BM25 score
  naturally gets split on more than a bounded [0,1] feature regardless of
  whether any split is actually decisive -- high importance isn't the same
  as separating the classes. top3_mean_dense is the real signal: only 18%
  overlap, rejected mean 0.33 vs correct mean 0.57 -- consistent with the
  original Raouliii finding from two sessions ago (strong top1 match, weak
  rank-2/3 support). n_candidates_above_floor reconfirmed dead beyond
  doubt: identical constant [6.0, 6.0] for BOTH groups now, not just the
  rejections. Two viable next experiments identified, neither attempted
  yet: decision-threshold tuning on the already-trained classifier
  (cheapest to test, no retraining needed), or a larger calibration set
  (already flagged, still untried). Both need the same real-embeddings
  verification as everything else here before being trusted.
- 2026-08-20 — Tested decision-threshold tuning: no improvement available.
  scripts/sweep_threshold.py swept 9 candidate thresholds against the
  already-trained real-embeddings classifier. Clean, monotonic trade-off
  curve -- explicitly checked, no threshold beats the current default (0.5)
  on both coverage AND guard simultaneously. Confirmed the hop-0-only
  approximation's gap against real numbers is a stable, expected pattern
  (reformulation adds ~coverage, costs ~guard), consistent across both
  embedders, not noise. Third hypothesis ruled out by real testing in this
  investigation -- the classifier isn't miscalibrated, it's making a
  reasonable trade-off given current features/data.
- 2026-08-20 — Fixed a real MemoryError on the actual machine running these
  evals (not the redundant-computation bug from before -- a different,
  deeper one: base cost of dense vectors accumulating across ~5 heavy
  tests in one pytest process). Reducing embedding dimensionality and
  shrinking the regression-test corpus were both tested and rejected, not
  just assumed risky. gc.collect() (tests/conftest.py) fixed 2 of 3
  failures, confirmed by real before/after output. The third needed a
  targeted fix specific to that test. 78 tests passing on the real machine.