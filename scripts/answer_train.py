"""Fine-tune the small cross-encoder to predict P(unanswerable | question, evidence); then evaluate on SQuAD2 validation and export int8 ONNX."""
import pandas, sklearn.metrics
import json, sys, time, pathlib, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
ROOT = pathlib.Path(__file__).resolve().parent.parent; W = ROOT / "guard_work"; OUT = W / "answer_model"
sys.path.insert(0, str(ROOT / "src" / "verityrag" / "guard")); sys.path.insert(0, str(ROOT / "scripts"))
from ctx import select_context
from guard_metrics import best_threshold, report
BASE, MAXLEN, BS, EPOCHS, LR, dev = "cross-encoder/nli-deberta-v3-xsmall", 256, 8, 1, 4e-5, "cuda"
load = lambda s: [json.loads(l) for l in open(W / f"answer_{s}.jsonl", encoding="utf-8")]
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2, ignore_mismatched_sizes=True).to(dev)
def enc(rows, device=True):
    e = tok([select_context(r["context"], r["question"], 110) for r in rows], [f"question: {r['question']}" for r in rows], truncation="only_first", max_length=MAXLEN, padding=True, return_tensors="pt")
    return {k: v.to(dev) for k, v in e.items()} if device else e
@torch.no_grad()
def predict(rows, bs=32):
    model.eval(); out = []; order = np.argsort([len(r["context"]) for r in rows])
    for i in range(0, len(rows), bs):
        with torch.autocast("cuda", dtype=torch.float16): lg = model(**enc([rows[j] for j in order[i:i + bs]])).logits.float()
        out.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    p = np.empty(len(rows)); p[order] = np.concatenate(out); return p
if __name__ == "__main__":
    tr, va, te = load("train"), load("val"), load("test")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01); steps = EPOCHS * (len(tr) // BS)
    sch = get_linear_schedule_with_warmup(opt, int(.06 * steps), steps); scaler = torch.amp.GradScaler(); t0 = time.time(); step = 0
    for ep in range(EPOCHS):
        model.train(); idx = np.random.RandomState(ep).permutation(len(tr))
        for i in range(0, len(idx) - BS + 1, BS):
            b = [tr[j] for j in idx[i:i + BS]]; y = torch.tensor([r["label"] for r in b]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16): loss = torch.nn.functional.cross_entropy(model(**enc(b)).logits.float(), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step(); step += 1
            if step % 20 == 0: print(f"ep{ep} step {step}/{steps} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    pv = predict(va); yv = np.array([r["label"] for r in va]); t = best_threshold(pv, yv); print("EPOCH 0 val", report(pv, yv, t), flush=True)
    pt = predict(te); yt = np.array([r["label"] for r in te]); res = {"all": report(pt, yt, t)}
    for k in ("answerable", "adversarial", "unrelated"):
        m = np.array([r["kind"] == k for r in te]); res[k] = {"flagged_unanswerable_rate": round(float((pt[m] >= t).mean()), 4), "n": int(m.sum())}
    OUT.mkdir(exist_ok=True); model.save_pretrained(OUT); tok.save_pretrained(OUT); json.dump({"threshold": t, "test": res}, open(OUT / "answer_meta.json", "w"), indent=1); print("TEST", json.dumps(res), flush=True)
    from onnxruntime.quantization import quantize_dynamic, QuantType
    m = AutoModelForSequenceClassification.from_pretrained(OUT).eval().cpu(); d = tok("a", "b", return_tensors="pt")
    torch.onnx.export(m, (d["input_ids"], d["attention_mask"]), str(OUT / "answer_fp32.onnx"), input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}}, opset_version=17, dynamo=False)
    quantize_dynamic(str(OUT / "answer_fp32.onnx"), str(OUT / "answer_int8.onnx"), weight_type=QuantType.QInt8); print("DONE exported int8", flush=True)
