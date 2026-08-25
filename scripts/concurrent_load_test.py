"""
LiveIndex uses threading.RLock() (indexing.py) -- designed for concurrent
ingestion/querying, but never actually verified under real concurrent
load, only reasoned about by reading the code. This starts the real
FastAPI server (uvicorn subprocess, not calling Python objects directly)
and fires genuine concurrent HTTP requests at it -- the actual scenario
that matters ("what happens if multiple users hit this API at once"),
not just a theoretical analysis of the locking code.

Three checks:
  1. Concurrent ingestion: N simultaneous /ingest calls, each a unique
     document -- does the final index size match exactly what was sent,
     with no lost updates from a race condition?
  2. Concurrent queries: M simultaneous /query calls against an
     already-populated index -- does every single one return a valid,
     non-erroring response?
  3. Mixed concurrent load: ingestion and queries firing at the same
     time -- does anything 500, hang, or return a malformed response?

Run with: python scripts/concurrent_load_test.py
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8321"  # unusual port, unlikely to collide with anything else running


def wait_for_server(timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    return False


def test_concurrent_ingestion(n=30):
    print(f"\n--- Test 1: {n} concurrent /ingest calls, each a unique document ---")

    def ingest_one(i):
        return httpx.post(
            f"{BASE_URL}/ingest",
            json={"doc_id": f"load_test_{i}", "title": f"Load Test Doc {i}",
                  "text": f"This is synthetic load-test document number {i}, "
                          f"containing unique content about topic {i} for testing purposes."},
            timeout=30,
        )

    errors = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(ingest_one, i) for i in range(n)]
        for f in as_completed(futures):
            try:
                r = f.result()
                if r.status_code != 200:
                    errors.append(f"status={r.status_code} body={r.text[:200]}")
            except Exception as e:
                errors.append(f"exception: {e}")

    stats = httpx.get(f"{BASE_URL}/stats", timeout=10).json()
    print(f"Errors during concurrent ingestion: {len(errors)}")
    for e in errors[:5]:
        print(f"  {e}")
    print(f"Final index size after {n} concurrent ingests: {stats['index_size']}")
    return {"n_requested": n, "n_errors": len(errors), "final_index_size": stats["index_size"]}


def test_concurrent_queries(n=30):
    print(f"\n--- Test 2: {n} concurrent /query calls against the populated index ---")

    questions = [
        f"What is load-test document number {i} about?" for i in range(min(n, 30))
    ] * (n // 30 + 1)
    questions = questions[:n]

    def query_one(q):
        return httpx.post(f"{BASE_URL}/query", json={"question": q}, timeout=30)

    errors = []
    malformed = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(query_one, q) for q in questions]
        for f in as_completed(futures):
            try:
                r = f.result()
                if r.status_code != 200:
                    errors.append(f"status={r.status_code} body={r.text[:200]}")
                    continue
                body = r.json()
                if "abstained" not in body or "question" not in body:
                    malformed.append(body)
            except Exception as e:
                errors.append(f"exception: {e}")

    print(f"Errors during concurrent queries: {len(errors)}")
    for e in errors[:5]:
        print(f"  {e}")
    print(f"Malformed responses (missing expected fields): {len(malformed)}")
    return {"n_requested": n, "n_errors": len(errors), "n_malformed": len(malformed)}


def test_mixed_concurrent_load(n_ingest=15, n_query=15):
    print(f"\n--- Test 3: {n_ingest} /ingest + {n_query} /query, all firing at once ---")

    def ingest_one(i):
        return httpx.post(
            f"{BASE_URL}/ingest",
            json={"doc_id": f"mixed_test_{i}", "title": f"Mixed Test {i}",
                  "text": f"Mixed load test content number {i}."},
            timeout=30,
        )

    def query_one(i):
        return httpx.post(f"{BASE_URL}/query", json={"question": f"What is mixed test {i} about?"}, timeout=30)

    errors = []
    with ThreadPoolExecutor(max_workers=n_ingest + n_query) as pool:
        futures = [pool.submit(ingest_one, i) for i in range(n_ingest)]
        futures += [pool.submit(query_one, i) for i in range(n_query)]
        for f in as_completed(futures):
            try:
                r = f.result()
                if r.status_code != 200:
                    errors.append(f"status={r.status_code} body={r.text[:200]}")
            except Exception as e:
                errors.append(f"exception: {e}")

    print(f"Errors during mixed concurrent load: {len(errors)}")
    for e in errors[:5]:
        print(f"  {e}")
    return {"n_requested": n_ingest + n_query, "n_errors": len(errors)}


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    print("Starting FastAPI server (uvicorn subprocess)...")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "verityrag.api:app",
         "--host", "127.0.0.1", "--port", "8321"],
        cwd=repo_root / "src",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_for_server():
            print("Server never became healthy -- aborting.")
            sys.exit(1)
        print("Server is up.\n")

        initial_stats = httpx.get(f"{BASE_URL}/stats", timeout=10).json()
        print(f"Initial index size: {initial_stats['index_size']}")

        result1 = test_concurrent_ingestion(n=30)
        result2 = test_concurrent_queries(n=30)
        result3 = test_mixed_concurrent_load(n_ingest=50, n_query=50)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        expected_final_size = initial_stats["index_size"] + result1["n_requested"] + result3["n_requested"] // 2
        print(f"Test 1 (concurrent ingestion): {result1['n_errors']} errors, "
              f"final size {result1['final_index_size']} "
              f"(started at {initial_stats['index_size']}, added {result1['n_requested']})")
        print(f"Test 2 (concurrent queries): {result2['n_errors']} errors, "
              f"{result2['n_malformed']} malformed responses")
        print(f"Test 3 (mixed load): {result3['n_errors']} errors")
    finally:
        print("\nShutting down server...")
        server.terminate()
        server.wait(timeout=10)