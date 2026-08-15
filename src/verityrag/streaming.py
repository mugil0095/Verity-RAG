"""
Real-time ingestion stream.

This is what makes "real-time" a testable, demonstrable property rather than
a claim: `LiveDocumentStream` runs a background thread that feeds documents
into the pipeline one at a time (as a webhook or Kafka consumer would),
instead of the whole corpus being bulk-loaded at process start. A query
issued mid-stream only sees whatever has been ingested so far, and a document
that arrives becomes retrievable within one ingestion cycle -- no restart,
no offline reindex job. `test_pipeline_integration.py` asserts this directly
(a doc is unsearchable before it streams in, and searchable shortly after).
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

from .pipeline import VerityRAGPipeline


@dataclass
class StreamStats:
    docs_ingested: int = 0
    chunks_ingested: int = 0
    is_running: bool = False


class LiveDocumentStream:
    def __init__(self, pipeline: VerityRAGPipeline, delay_seconds: float = 0.05):
        self.pipeline = pipeline
        self.delay_seconds = delay_seconds
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self.stats = StreamStats()

    def enqueue(self, doc: dict):
        """Push a document onto the live stream (e.g. from a webhook handler)."""
        self._queue.put(doc)

    def enqueue_many(self, docs: list[dict]):
        for d in docs:
            self._queue.put(d)

    def _run(self):
        self.stats.is_running = True
        while not self._stop_flag.is_set():
            try:
                doc = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            n_chunks = self.pipeline.ingest_document(doc["doc_id"], doc["title"], doc["text"])
            self.stats.docs_ingested += 1
            self.stats.chunks_ingested += n_chunks
            time.sleep(self.delay_seconds)
        self.stats.is_running = False

    def start(self):
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0):
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def wait_until_ingested(self, n_docs: int, timeout: float = 5.0) -> bool:
        """Poll helper for tests/demo: block until at least n_docs have been ingested."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.stats.docs_ingested >= n_docs:
                return True
            time.sleep(0.02)
        return self.stats.docs_ingested >= n_docs
