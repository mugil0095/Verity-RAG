"""Print the README results table from guard_work/baseline_results.json + guard_results.json."""
import json, pathlib
W = pathlib.Path(__file__).resolve().parent.parent / "guard_work"
b, g = json.load(open(W / "baseline_results.json")), json.load(open(W / "guard_results.json"))
rows = [("Heuristic grounding (original)", b["all"], b["ragtruth"], b["halueval"], "-", "-")]
for n, k in (("Verity-Guard fp32 (ONNX)", "fp32"), ("Verity-Guard int8 (ONNX)", "int8")):
    rows.append((n, g[k]["all"], g[k]["ragtruth"], g[k]["halueval"], g[k]["latency_ms"]["p50"], g[k]["latency_ms"]["p95"]))
print("| Method | AUROC | F1 | Precision | Recall | ECE | RAGTruth F1 | HaluEval F1 | CPU p50 ms | p95 ms |\n|---|---|---|---|---|---|---|---|---|---|")
for n, a, r, h, p50, p95 in rows:
    print(f"| {n} | {a['auroc']} | {a['f1']} | {a['precision']} | {a['recall']} | {a['ece']} | {r['f1']} | {h['f1']} | {p50} | {p95} |")
