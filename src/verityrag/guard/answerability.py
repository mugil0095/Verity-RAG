"""Answerability gate: P(question is NOT answerable from the retrieved evidence), from a small fine-tuned
cross-encoder (int8 ONNX, CPU). Runs before generation, next to the LightGBM sufficiency gate."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .ctx import select_context

DEFAULT_DIR = Path(os.environ.get("VERITY_ANSWER_DIR", Path(__file__).resolve().parents[3] / "guard_work" / "answer_model"))


class AnswerabilityGate:
    def __init__(self, model_dir: str | Path | None = None, threshold: float | None = None, threads: int = 4):
        from onnxruntime import InferenceSession, SessionOptions
        from transformers import AutoTokenizer

        d = Path(model_dir or DEFAULT_DIR)
        self.tok = AutoTokenizer.from_pretrained(d)
        self.threshold = threshold if threshold is not None else json.load(open(d / "answer_meta.json"))["threshold"]
        so = SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = InferenceSession(str(d / "answer_int8.onnx"), so, providers=["CPUExecutionProvider"])

    def p_unanswerable(self, question: str, evidence: list | str) -> float:
        ctx = evidence if isinstance(evidence, str) else " ".join(getattr(e, "text", str(e)) for e in evidence)
        e = self.tok(select_context(ctx, question, 110), f"question: {question}", truncation="only_first", max_length=256, return_tensors="np")
        lg = self.sess.run(None, {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0]
        ex = np.exp(lg - lg.max(-1, keepdims=True))
        return float((ex / ex.sum(-1, keepdims=True))[0, 1])

    def is_answerable(self, question: str, evidence: list | str) -> bool:
        return self.p_unanswerable(question, evidence) < self.threshold
