"""Build response-level hallucination datasets for Verity-Guard.
Label 1 = hallucinated (unsupported by the context), 0 = grounded.
Sources: RAGTruth (processed, official train/test) + HaluEval QA (10k, split 80/10/10 by question)."""
import json, random, ast, pathlib
from datasets import load_dataset

OUT = pathlib.Path(__file__).resolve().parent.parent / "guard_work"
OUT.mkdir(exist_ok=True)
random.seed(0)

def ragtruth(split):
    d = load_dataset("wandb/RAGTruth-processed")[split]
    rows = []
    for r in d:
        lp = r["hallucination_labels_processed"]
        lp = ast.literal_eval(lp) if isinstance(lp, str) else lp
        halluc = int(lp["evident_conflict"] > 0 or lp["baseless_info"] > 0)
        rows.append({"src": "ragtruth", "task": r["task_type"], "context": r["context"],
                     "query": r["query"], "answer": r["output"], "label": halluc})
    return rows

def halu():
    d = load_dataset("pminervini/HaluEval", "qa_samples")["data"]
    rows = [{"src": "halueval", "task": "QA", "context": r["knowledge"], "query": r["question"],
             "answer": r["answer"], "label": int(r["hallucination"] == "yes")} for r in d]
    qs = sorted({r["query"] for r in rows}); random.shuffle(qs)
    n = len(qs); tr, va = set(qs[: int(.8 * n)]), set(qs[int(.8 * n): int(.9 * n)])
    sp = {"train": [], "val": [], "test": []}
    for r in rows:
        sp["train" if r["query"] in tr else "val" if r["query"] in va else "test"].append(r)
    return sp

rt_train, rt_test = ragtruth("train"), ragtruth("test")
random.shuffle(rt_train); k = int(.9 * len(rt_train))
h = halu()
splits = {"train": rt_train[:k] + h["train"], "val": rt_train[k:] + h["val"], "test": rt_test + h["test"]}
for s, rows in splits.items():
    random.shuffle(rows)
    with open(OUT / f"{s}.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    pos = sum(r["label"] for r in rows)
    print(s, len(rows), "hallucinated:", pos, f"({pos/len(rows):.1%})", {x: sum(r['src'] == x for r in rows) for x in ("ragtruth", "halueval")})
