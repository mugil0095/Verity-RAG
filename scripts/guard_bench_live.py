"""Live int8 benchmark: streams running metrics to guard_work/bench.json (shown on the dashboard) and writes guard_results.json at the end."""
import pandas, sklearn.metrics
import json, sys, time, pathlib, numpy as np
from sklearn.metrics import roc_auc_score
ROOT = pathlib.Path(__file__).resolve().parent.parent; W = ROOT / "guard_work"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from verityrag.guard import LearnedGuard
from guard_metrics import report
g = LearnedGuard(W / "model", threads=4); T = g.threshold
rows = [json.loads(l) for l in open(W / "test.jsonl", encoding="utf-8")]; y = np.array([r["label"] for r in rows])
p, lat = np.zeros(len(rows)), []; t0 = time.time()
for i, r in enumerate(rows):
    v = g.score(r["answer"], r["context"], r["query"]); p[i] = v.p_hallucinated; lat.append(v.latency_ms)
    if (i + 1) % 10 == 0 or i + 1 == len(rows):
        n = i + 1; yy = y[:n]; pr = p[:n] >= T
        tp = int((pr & (yy == 1)).sum()); prec = tp / max(1, int(pr.sum())); rec = tp / max(1, int((yy == 1).sum()))
        json.dump({"done": n, "total": len(rows), "acc": round(float((pr == yy).mean()), 4), "precision": round(prec, 4), "recall": round(rec, 4),
                   "f1": round(2 * prec * rec / max(1e-9, prec + rec), 4), "p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1),
                   "per_s": round(n / (time.time() - t0), 2), "eta_s": int((len(rows) - n) / (n / (time.time() - t0))), "finished": n == len(rows)}, open(W / "bench.json", "w"))
out = {"threshold": T, "size_mb": {"int8": round((W / "model" / "guard_int8.onnx").stat().st_size / 1e6, 1)},
       "int8": {"all": report(p, y, T), **{s: report(p[[r["src"] == s for r in rows]], y[[r["src"] == s for r in rows]], T) for s in ("ragtruth", "halueval")},
                "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1)}}}
json.dump(out, open(W / "guard_results.json", "w"), indent=1); print("int8", json.dumps(out["int8"]["all"]), out["int8"]["latency_ms"])
