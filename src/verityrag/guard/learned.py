"""Verity-Guard: a fine-tuned small cross-encoder (int8 ONNX, CPU) that scores whether an answer is
supported by its retrieved evidence. Drop-in second opinion next to the heuristic grounding checker
in grounding.py. Model dir holds guard_int8.onnx, tokenizer files and guard_meta.json (threshold)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ctx import select_context

DEFAULT_DIR = Path(os.environ.get("VERITY_GUARD_DIR", Path(__file__).resolve().parents[3] / "guard_work" / "model"))


@dataclass
class GuardVerdict:
    p_hallucinated: float
    hallucinated: bool
    verdict: str  # "supported" | "unsupported" | "uncertain"
    latency_ms: float


class LearnedGuard:
    def __init__(self, model_dir: str | Path | None = None, uncertain_band: float = 0.15, threads: int = 4):
        from onnxruntime import InferenceSession, SessionOptions
        from transformers import AutoTokenizer

        d = Path(model_dir or DEFAULT_DIR)
        self.tok = AutoTokenizer.from_pretrained(d)
        self.threshold = json.load(open(d / "guard_meta.json"))["threshold"]
        self.band = uncertain_band
        so = SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = InferenceSession(str(d / "guard_int8.onnx"), so, providers=["CPUExecutionProvider"])

    def score(self, answer: str, evidence: list | str, question: str = "") -> GuardVerdict:
        import time

        ctx = evidence if isinstance(evidence, str) else " ".join(getattr(e, "text", str(e)) for e in evidence)
        e = self.tok(select_context(ctx, answer), f"question: {question} answer: {answer}", truncation="longest_first", max_length=320, return_tensors="np")
        t0 = time.perf_counter()
        lg = self.sess.run(None, {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0]
        ms = (time.perf_counter() - t0) * 1000
        ex = np.exp(lg - lg.max(-1, keepdims=True))
        p = float((ex / ex.sum(-1, keepdims=True))[0, 1])
        if abs(p - self.threshold) < self.band * min(self.threshold, 1 - self.threshold) * 2:
            v = "uncertain"
        else:
            v = "unsupported" if p >= self.threshold else "supported"
        return GuardVerdict(round(p, 4), p >= self.threshold, v, round(ms, 1))
