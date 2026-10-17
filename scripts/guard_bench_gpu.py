"""GPU benchmark (fp16 PyTorch, CUDA): batch-1 latency + accuracy on the full test set, then batched throughput. Streams to guard_work/bench_gpu.json."""
import pandas, sklearn.metrics
import json, sys, time, pathlib, numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parent.parent; W = ROOT / "guard_work"
sys.path.insert(0, str(ROOT / "src" / "verityrag" / "guard")); sys.path.insert(0, str(ROOT / "scripts"))
from ctx import select_context
from guard_metrics import report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(W / "model"); T = json.load(open(W / "model" / "guard_meta.json"))["threshold"]
m = AutoModelForSequenceClassification.from_pretrained(W / "model").half().cuda().eval()
rows = [json.loads(l) for l in open(W / "test.jsonl", encoding="utf-8")]; y = np.array([r["label"] for r in rows])
def enc(b): return {k: v.cuda() for k, v in tok([select_context(r["context"], r["answer"]) for r in b], [f"question: {r['query']} answer: {r['answer']}" for r in b], truncation="longest_first", max_length=320, padding=True, return_tensors="pt").items()}
@torch.no_grad()
def run(b):
    e = enc(b); torch.cuda.synchronize(); t = time.perf_counter(); lg = m(**e).logits.float(); torch.cuda.synchronize()
    return torch.softmax(lg, -1)[:, 1].cpu().numpy(), (time.perf_counter() - t) * 1000
for _ in range(10): run(rows[:1])  # warm-up
p, lat, t0 = np.zeros(len(rows)), [], time.time(); live = lambda **k: json.dump(k, open(W / "bench_gpu.json", "w"))
for i, r in enumerate(rows):
    pp, ms = run([r]); p[i] = pp[0]; lat.append(ms)
    if (i + 1) % 25 == 0 or i + 1 == len(rows):
        n = i + 1; pr = p[:n] >= T; yy = y[:n]; tp = int((pr & (yy == 1)).sum()); pre = tp / max(1, int(pr.sum())); rec = tp / max(1, int((yy == 1).sum()))
        live(done=n, total=len(rows), phase="batch-1 latency", acc=round(float((pr == yy).mean()), 4), f1=round(2 * pre * rec / max(1e-9, pre + rec), 4), p50=round(float(np.percentile(lat, 50)), 1), p95=round(float(np.percentile(lat, 95)), 1), per_s=round(n / (time.time() - t0), 1), finished=False)
order = np.argsort([len(r["context"]) + len(r["answer"]) for r in rows]); t1 = time.time()
for i in range(0, len(rows), 32): run([rows[j] for j in order[i:i + 32]])
thr = len(rows) / (time.time() - t1)
out = {"fp16_gpu": {"all": report(p, y, T), "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1)}, "throughput_batch32_per_s": round(thr, 1), "gpu": torch.cuda.get_device_name(0)}}
json.dump(out, open(W / "guard_gpu_results.json", "w"), indent=1)
live(done=len(rows), total=len(rows), phase="done", acc=out["fp16_gpu"]["all"]["acc"], f1=out["fp16_gpu"]["all"]["f1"], p50=out["fp16_gpu"]["latency_ms"]["p50"], p95=out["fp16_gpu"]["latency_ms"]["p95"], per_s=round(thr, 1), finished=True)
print("gpu", json.dumps(out["fp16_gpu"]["all"]), out["fp16_gpu"]["latency_ms"], "batch32/s", round(thr, 1))
