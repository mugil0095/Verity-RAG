"""Build in-distribution guard data: real LLM answers (Qwen2.5-0.5B) over SQuAD 2.0 evidence, labelled by gold-answer presence.
label 0 = answer contains a gold answer string (supported, evidence includes the gold paragraph); label 1 = it does not (wrong / unsupported / refusal).
Noisy: correct paraphrases count as label 1. Train/val articles come from SQuAD2 train; test from SQuAD2 validation (different articles)."""
import pandas, sklearn.metrics
import json, random, re, sys, time, pathlib, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
W = pathlib.Path(__file__).resolve().parent.parent / "guard_work"; ID = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(ID, local_files_only=True, padding_side="left")
lm = AutoModelForCausalLM.from_pretrained(ID, local_files_only=True, torch_dtype=torch.float16).cuda().eval()
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower())
def build(split, n, seed):
    d = list(load_dataset("rajpurkar/squad_v2", split=split)); rnd = random.Random(seed)
    by = {}
    for r in d: by.setdefault(r["title"], []).append(r["context"])
    ans = [r for r in d if r["answers"]["text"]]; rnd.shuffle(ans); items = []
    for r in ans[:n]:
        others = [c for c in set(by[r["title"]]) if c != r["context"]]; rnd.shuffle(others)
        ev = [r["context"]] + others[:1]; rnd.shuffle(ev); items.append((r, ev))
    return items
@torch.no_grad()
def gen(prompts, bs=24):
    out = []
    for i in range(0, len(prompts), bs):
        msgs = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for p in prompts[i:i + bs]]
        e = tok(msgs, return_tensors="pt", padding=True).to("cuda"); o = lm.generate(**e, max_new_tokens=48, do_sample=False)
        out += [tok.decode(x[e["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]
        if (i // bs) % 10 == 0: print(f"gen {i + len(msgs)}/{len(prompts)} {time.time() - t0:.0f}s", flush=True)
    return out
for name, split, n, seed in (("llm_train", "train", 1600, 11), ("llm_val", "train", 250, 12), ("llm_test", "validation", 400, 13)):
    t0 = time.time(); items = build(split, n, seed)
    prompts = ["Answer the question using ONLY the evidence below. If the evidence is insufficient, say so explicitly.\n\nEvidence:\n" + "\n\n".join(f"[c{i}] {c}" for i, c in enumerate(ev)) + f"\n\nQuestion: {r['question']}\nAnswer:" for r, ev in items]
    answers = gen(prompts); rows = []
    for (r, ev), a in zip(items, answers):
        hit = any(norm(g) in norm(a) for g in r["answers"]["text"])
        rows.append({"src": "llm", "task": "QA", "context": " ".join(ev), "query": r["question"], "answer": a, "label": int(not hit)})
    if name == "llm_val":
        tr = {json.loads(l)["query"] for l in open(W / "llm_train.jsonl", encoding="utf-8")}; rows = [x for x in rows if x["query"] not in tr]
    with open(W / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for x in rows: f.write(json.dumps(x) + "\n")
    print("SAVED", name, len(rows), "label1 (gold missing):", sum(x["label"] for x in rows), flush=True)
print("ALLDONE", flush=True)
