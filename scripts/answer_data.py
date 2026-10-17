"""Answerability data from SQuAD 2.0 (train articles are disjoint from the SQuAD 1.1 dev corpus used by eval.py).
label 1 = UNANSWERABLE from this evidence (adversarial SQuAD2 question, or a question paired with an unrelated paragraph); 0 = answerable."""
import json, random, pathlib
from datasets import load_dataset
W = pathlib.Path(__file__).resolve().parent.parent / "guard_work"; rnd = random.Random(0)
def build(split, n, seed):
    d = list(load_dataset("rajpurkar/squad_v2", split=split)); rnd = random.Random(seed); rnd.shuffle(d)
    ans = [r for r in d if r["answers"]["text"]]; una = [r for r in d if not r["answers"]["text"]]
    rows = []
    for r in ans[: n // 2]:
        rows.append({"question": r["question"], "context": r["context"], "label": 0, "kind": "answerable"})
    for r in una[: n // 4]:
        rows.append({"question": r["question"], "context": r["context"], "label": 1, "kind": "adversarial"})
    for r in ans[n // 2: n // 2 + n // 4]:
        o = rnd.choice(d)
        if o["title"] != r["title"]: rows.append({"question": r["question"], "context": o["context"], "label": 1, "kind": "unrelated"})
    rnd.shuffle(rows); return rows
for name, split, n, seed in (("train", "train", 12000, 1), ("val", "train", 1500, 2), ("test", "validation", 2400, 3)):
    rows = build(split, n, seed)
    if name == "val":  # keep val articles apart from train rows by question text
        tr = {json.loads(l)["question"] for l in open(W / "answer_train.jsonl", encoding="utf-8")}; rows = [r for r in rows if r["question"] not in tr]
    with open(W / f"answer_{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(name, len(rows), "unanswerable:", sum(r["label"] for r in rows))
