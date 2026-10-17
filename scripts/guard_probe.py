"""Fluent-answer probe: held-out HaluEval test contexts, answers written as full sentences like Verity-RAG's extractive generator.
label 0 = the supporting sentence; label 1 = that sentence with numbers altered, or a sentence from an unrelated example."""
import pandas, sklearn.metrics
import json, sys, pathlib, random, re, numpy as np
from sklearn.metrics import roc_auc_score
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
src = open(ROOT / "scripts" / "guard_train.py", encoding="utf-8").read()
ns = {}; exec(src[src.index("import random, re"):src.index("def enc(rows):")], ns)
from verityrag.guard import LearnedGuard
g = LearnedGuard(ROOT / "guard_work" / "model"); T = g.threshold
rows = [json.loads(l) for l in open(ROOT / "guard_work" / "test.jsonl", encoding="utf-8")]
pos = [r for r in rows if r["src"] == "halueval" and r["label"] == 0]
rnd = random.Random(7); items = []
for r in pos[:300]:
    s = ns["_support"](r["context"], r["answer"])
    if not s: continue
    items.append((r["context"], r["query"], s, 0))
    if ns["_NUM"].search(s): items.append((r["context"], r["query"], ns["_NUM"].sub(lambda m: str(int(m.group()) + rnd.choice([-7, -3, 3, 11, 40])), s), 1))
    o = rnd.choice(pos); fs = ns["_support"](o["context"], o["answer"])
    if fs and o["context"] != r["context"]: items.append((r["context"], r["query"], fs, 1))
p = np.array([g.score(a, c, q).p_hallucinated for c, q, a, _ in items]); y = np.array([l for *_, l in items])
out = {"n": len(y), "auroc": round(float(roc_auc_score(y, p)), 4), "acc": round(float(((p >= T) == y).mean()), 4),
       "supported_flagged_rate": round(float((p[y == 0] >= T).mean()), 4), "unsupported_caught_rate": round(float((p[y == 1] >= T).mean()), 4)}
print(json.dumps(out)); json.dump(out, open(ROOT / "guard_work" / "probe_results.json", "w"))
