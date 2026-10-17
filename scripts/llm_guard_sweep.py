"""Run the local-LLM SQuAD eval ONCE, record the guard's p(hallucinated) for every generated answer, then sweep guard thresholds offline."""
import pandas, sklearn.metrics
import json, sys, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from transformers import AutoTokenizer, AutoModelForCausalLM
from verityrag.eval import run_eval, _load
from verityrag.generation import LLMGenerator
from verityrag.guard import LearnedGuard
from verityrag.guard.learned import GuardVerdict
ID = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(ID, local_files_only=True); lm = AutoModelForCausalLM.from_pretrained(ID, local_files_only=True, torch_dtype=torch.float16).cuda().eval()
@torch.no_grad()
def complete(prompt):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    out = lm.generate(**ids, max_new_tokens=80, do_sample=False); return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
class Rec(LearnedGuard):
    log = []
    def score(self, answer, evidence, question=""):
        v = super().score(answer, evidence, question); Rec.log.append({"q": question, "a": answer, "p": v.p_hallucinated}); return GuardVerdict(v.p_hallucinated, False, "supported", v.latency_ms)
run_eval(verbose=False, generator=LLMGenerator(complete_fn=complete), answer_guard=Rec())
gold = {x["question"]: x["gold_answer"] for x in _load("eval_answerable.json")}; una = {x["question"] for x in _load("eval_unanswerable.json")}
recs = [dict(r, kind="answerable" if r["q"] in gold else "unanswerable" if r["q"] in una else "calib", hit=(r["q"] in gold and gold[r["q"]].lower() in r["a"].lower())) for r in Rec.log]
json.dump(recs, open(ROOT / "guard_work" / "llm_guard_records.json", "w"), indent=1)
test = [r for r in recs if r["kind"] != "calib"]; A = [r for r in test if r["kind"] == "answerable"]; U = [r for r in test if r["kind"] == "unanswerable"]
print("SWEEP answered_answerable=%d (correct %d, wrong %d), answered_unanswerable(leaks)=%d" % (len(A), sum(r["hit"] for r in A), sum(not r["hit"] for r in A), len(U)))
for t in (0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 1.01):
    kept = [r for r in A if r["p"] < t]; print("SWEEP t=%.3f kept_correct=%d kept_wrong=%d leaks_kept=%d" % (t, sum(r["hit"] for r in kept), sum(not r["hit"] for r in kept), sum(r["p"] < t for r in U)))
for r in [r for r in A if r["p"] >= 0.5 and r["hit"]][:4]: print("FLAGGED-CORRECT", round(r["p"], 3), "|", r["q"][:70], "|", r["a"][:140].replace("\n", " "))
for r in [r for r in A if r["p"] >= 0.5 and not r["hit"]][:4]: print("FLAGGED-WRONG", round(r["p"], 3), "|", r["q"][:70], "|", r["a"][:140].replace("\n", " "))
print("ALLDONE")
