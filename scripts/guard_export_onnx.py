"""Export the fine-tuned guard to ONNX, quantize to int8, then benchmark accuracy + CPU latency of fp32 vs int8 vs the original heuristic."""
import pandas, sklearn.metrics
import json, sys, time, pathlib, numpy as np, torch
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'src' / 'verityrag' / 'guard'))
from ctx import select_context
from guard_metrics import report
W = pathlib.Path(__file__).resolve().parent.parent / "guard_work"; import os
M = W / os.environ.get("GUARD_OUT", "model")
tok = AutoTokenizer.from_pretrained(M); meta = json.load(open(M / "guard_meta.json")); T = meta["threshold"]
fp32, int8 = M / "guard_fp32.onnx", M / "guard_int8.onnx"
if not fp32.exists():
    m = AutoModelForSequenceClassification.from_pretrained(M).eval().cpu()
    d = tok("a", "b", return_tensors="pt")
    torch.onnx.export(m, (d["input_ids"], d["attention_mask"]), str(fp32), input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}}, opset_version=17, dynamo=False)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
rows = [json.loads(l) for l in open(W / "test.jsonl", encoding="utf-8")]
rows = rows[: int(os.environ.get("TEST_N", len(rows)))] if (os := __import__("os")) else rows
y = np.array([r["label"] for r in rows])
def enc(b): return tok([select_context(r["context"], r["answer"]) for r in b], [f"question: {r['query']} answer: {r['answer']}" for r in b], truncation="longest_first", max_length=320, padding=True, return_tensors="np")
def run(path, rows, bs=1):
    so = SessionOptions(); so.intra_op_num_threads = 4; s = InferenceSession(str(path), so, providers=["CPUExecutionProvider"]); ps, lat = [], []
    for i in range(0, len(rows), bs):
        e = enc(rows[i:i + bs]); t0 = time.perf_counter()
        lg = s.run(None, {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0]
        lat.append((time.perf_counter() - t0) * 1000); ex = np.exp(lg - lg.max(-1, keepdims=True)); ps.extend((ex / ex.sum(-1, keepdims=True))[:, 1])
    return np.array(ps), np.array(lat)
out = {"threshold": T, "size_mb": {"fp32": round(fp32.stat().st_size / 1e6, 1), "int8": round(int8.stat().st_size / 1e6, 1)}}
import os
for name, path in ((("fp32", fp32),) if not os.environ.get("SKIP_FP32") else ()) + (("int8", int8),):
    p, lat = run(path, rows)  # batch size 1 = per-request serving latency
    out[name] = {"all": report(p, y, T), **{s: report(p[[r["src"] == s for r in rows]], y[[r["src"] == s for r in rows]], T) for s in ("ragtruth", "halueval")},
                 "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1)}}
    if name == "int8": np.save(W / "int8_test_p.npy", p)
    print(name, json.dumps(out[name]["all"]), out[name]["latency_ms"], flush=True)
json.dump(out, open(W / "guard_results.json", "w"), indent=1)
