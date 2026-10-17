"""SQuAD eval with a REAL (small, local) LLM generator -- Qwen2.5-0.5B-Instruct on the GPU, no API key -- so the guard sees genuine fabrications.
Configs: baseline (heuristic grounding only) | + learned guard | + guard + answerability."""
import pandas, sklearn.metrics
import json, sys, time, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
from transformers import AutoTokenizer, AutoModelForCausalLM
from verityrag.eval import run_eval
from verityrag.generation import LLMGenerator
from verityrag.guard import LearnedGuard, AnswerabilityGate
ID = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(ID, local_files_only=True); lm = AutoModelForCausalLM.from_pretrained(ID, local_files_only=True, torch_dtype=torch.float16).cuda().eval()
@torch.no_grad()
def complete(prompt: str) -> str:
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    out = lm.generate(**ids, max_new_tokens=80, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
gen = LLMGenerator(complete_fn=complete); guard = LearnedGuard(); ans = AnswerabilityGate(threshold=0.9); results = {}
for name, kw in (("baseline", {}), ("+guard", {"answer_guard": guard}), ("+guard +answerability(0.9)", {"answer_guard": guard, "answerability": ans})):
    t0 = time.time(); r = run_eval(verbose=False, generator=gen, **kw); r["wall_s"] = round(time.time() - t0)
    results[name] = {k: r[k] for k in ("attempted", "wrongly_abstained", "coverage_rate", "wrong_answers_among_attempted", "keyword_hit_rate_on_attempted", "avg_grounding_score_on_attempted", "hallucination_guard_rate", "abstain_reasons_on_answerable", "abstain_reasons_on_unanswerable", "latency_p50_ms", "wall_s")}
    print("RESULT", name, json.dumps(results[name]), flush=True)
json.dump(results, open(ROOT / "guard_work" / "squad_llm_results.json", "w"), indent=1); print("ALLDONE", flush=True)
