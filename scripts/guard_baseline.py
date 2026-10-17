"""Score the ORIGINAL Verity-RAG grounding checker (hashing embeddings + lexical) on the same test set."""
import json, sys, pathlib, types, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src")); sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verityrag.embedding import HashingEmbedder
from verityrag.grounding import check_grounding
from guard_metrics import best_threshold, report
W = pathlib.Path(__file__).resolve().parent.parent / "guard_work"
emb = HashingEmbedder()
def run(split):
    rows = [json.loads(l) for l in open(W / f"{split}.jsonl", encoding="utf-8")]
    p = np.array([1 - check_grounding(r["answer"], [types.SimpleNamespace(text=r["context"], chunk_id="c")], emb).overall_score for r in rows])
    return p, np.array([r["label"] for r in rows]), rows
if __name__ == "__main__":
    pv, yv, _ = run("val"); t = best_threshold(pv, yv)
    pt, yt, rows = run("test"); out = {"all": report(pt, yt, t)}
    for s in ("ragtruth", "halueval"):
        m = np.array([r["src"] == s for r in rows]); out[s] = report(pt[m], yt[m], t)
    np.save(W / "baseline_test_p.npy", pt); json.dump(out, open(W / "baseline_results.json", "w"), indent=1); print(json.dumps(out, indent=1))
