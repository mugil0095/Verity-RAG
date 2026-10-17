"""Fine-tune a small NLI cross-encoder as a response-level hallucination detector.
premise = retrieved context, hypothesis = question + answer. Output P(hallucinated)."""
import pandas, sklearn.metrics
import json, sys, time, pathlib, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'src' / 'verityrag' / 'guard'))
from ctx import select_context
from guard_metrics import best_threshold, report

W = pathlib.Path(__file__).resolve().parent.parent / "guard_work"
import os
OUT = W / os.environ.get("GUARD_OUT", "model")
MIX = os.environ.get("GUARD_MIX") == "1"
BASE, MAXLEN, BS, EPOCHS, LR = "cross-encoder/nli-deberta-v3-xsmall", 320, 8, 1, 4e-5
TRAIN_N, VAL_N = 10000, 1200
dev = "cuda"
load = lambda s: [json.loads(l) for l in open(W / f"{s}.jsonl", encoding="utf-8")]
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2, ignore_mismatched_sizes=True).to(dev)


import random, re
_SENT = re.compile(r"(?<=[.!?])\s+")
_NUM = re.compile(r"\d+")
def _support(ctx, ans):
    ws = set(re.findall(r"[a-z0-9]+", ans.lower()))
    ss = [x for x in _SENT.split(ctx) if 4 <= len(x.split()) <= 45]
    if not ss or not ws: return None
    best = max(ss, key=lambda x: len(ws & set(re.findall(r"[a-z0-9]+", x.lower()))) / (1 + 0.01 * len(x.split())))
    return best if len(ws & set(re.findall(r"[a-z0-9]+", best.lower()))) >= max(1, len(ws) // 2) else None
def augment(rows, seed=0):
    """Fix style shortcuts: add extractive-sentence answers like Verity-RAG's own generator produces.
    label 0: the context sentence that supports the answer; label 1: that sentence with numbers altered, or a sentence from an unrelated example."""
    rnd = random.Random(seed); T = ["The answer is {a}.", "According to the passage, {a}.", "It is {a}."]
    pool = [r for r in rows if r["src"] == "halueval" and r["label"] == 0]
    out = []
    for r in rows:
        if r["src"] != "halueval":
            out.append(r); continue
        if rnd.random() < 0.3:
            a = r["answer"].rstrip("."); r = dict(r, answer=rnd.choice(T).format(a=a))
        out.append(r)
        if r["label"] == 0 and rnd.random() < 0.7:
            sent = _support(r["context"], r["answer"])
            if sent:
                out.append(dict(r, answer=sent, label=0))
                if _NUM.search(sent) and rnd.random() < 0.6:
                    out.append(dict(r, answer=_NUM.sub(lambda m: str(int(m.group()) + rnd.choice([-7, -3, 3, 11, 40])), sent), label=1))
                else:
                    o = rnd.choice(pool); fs = _support(o["context"], o["answer"])
                    if fs and o["context"] != r["context"]: out.append(dict(r, answer=fs, label=1))
    rnd.shuffle(out); return out

def enc(rows):
    return tok([select_context(r["context"], r["answer"]) for r in rows], [f"question: {r['query']} answer: {r['answer']}" for r in rows],
               truncation="longest_first", max_length=MAXLEN, padding=True, return_tensors="pt")

@torch.no_grad()
def predict(rows, bs=32):
    model.eval(); out = []
    order = np.argsort([len(r["context"]) + len(r["answer"]) for r in rows])
    for i in range(0, len(rows), bs):
        b = [rows[j] for j in order[i:i + bs]]
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(**{k: v.to(dev) for k, v in enc(b).items()}).logits.float()
        out.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    p = np.empty(len(rows)); p[order] = np.concatenate(out); return p

if __name__ == "__main__":
    tr, va = augment(load("train")[:TRAIN_N])[:TRAIN_N], augment(load("val")[:VAL_N], 1)
    if MIX:
        ll = lambda n: [json.loads(l) for l in open(W / f"{n}.jsonl", encoding="utf-8")]
        tr = tr[:6000] + ll("llm_train"); va = va[:700] + ll("llm_val"); np.random.RandomState(5).shuffle(tr); print("MIX train", len(tr), "val", len(va), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * (len(tr) // BS)
    sch = get_linear_schedule_with_warmup(opt, int(.06 * steps), steps)
    scaler = torch.amp.GradScaler(); best = -1; t0 = time.time(); step = 0
    for ep in range(EPOCHS):
        model.train(); idx = np.random.RandomState(ep).permutation(len(tr))
        for i in range(0, len(idx) - BS + 1, BS):
            b = [tr[j] for j in idx[i:i + BS]]
            y = torch.tensor([r["label"] for r in b]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(**{k: v.to(dev) for k, v in enc(b).items()}).logits.float(), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step(); step += 1
            if step % 20 == 0: print(f"ep{ep} step {step}/{steps} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        p = predict(va); y = np.array([r["label"] for r in va]); t = best_threshold(p, y); rep = report(p, y, t)
        print(f"EPOCH {ep} val {rep}", flush=True)
        if rep["f1"] > best:
            best = rep["f1"]; model.save_pretrained(OUT); tok.save_pretrained(OUT)
            json.dump({"threshold": t, "val": rep}, open(OUT / "guard_meta.json", "w"))
    print("DONE best val f1", best)
