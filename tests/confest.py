"""
Autouse fixture forcing garbage collection after every test.

Context: on a memory-constrained machine, running the full test suite in
one pytest process hit real MemoryError crashes -- not the redundant-
computation bug fixed earlier (that's still fixed), but the base cost of
HashingEmbedder's dense 65536-dim vectors accumulating across many heavy
tests in one process (test_app.py alone does ~5 separate full-corpus
loads across its test functions). Reducing embedding dimensionality was
tested directly and rejected -- it measurably changes eval numbers (96%
coverage -> 92% at 16384 features, worse at smaller sizes), which would
silently invalidate every documented result in this project. Shrinking
test_eval_regression.py's corpus was also tested and rejected -- at 150
docs the reformulation-drift bug becomes nearly undetectable (0.200 vs
0.050, versus 0.4 vs 0.0 at the current 250-doc size), which would weaken
the exact regression protection that test exists for.

This is a genuinely honest, unverified-by-me mitigation, not a confirmed
fix: gc.collect() reclaims Python-level references, but whether that
translates into the OS actually reclaiming memory depends on the platform
allocator, which I can't test myself (this sandbox has more memory
headroom than the machine that hit the crash). Worth trying because it's
safe and costs nothing -- doesn't change any test's logic or assertions --
but if it doesn't fully resolve the crashes, the real fix is keeping
embeddings sparse throughout instead of densifying them (embedding.py's
`.toarray()` call), which needs its own dedicated, carefully-tested pass
rather than being rushed in response to a test failure.
"""
import gc

import pytest


@pytest.fixture(autouse=True)
def _gc_after_each_test():
    yield
    gc.collect()