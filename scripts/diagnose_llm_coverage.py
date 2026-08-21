"""
Diagnoses wrongly-abstained answerable questions when using a real LLM
generator: did the question never reach the LLM at all (rejected by the
sufficiency gate, same as it would be with any generator), or did the LLM
generate an answer that then failed the post-generation grounding check?
These need completely different explanations -- a gate rejection is about
retrieval/calibration, unrelated to the generator; a grounding rejection
means the LLM's own phrasing didn't score as well-supported by the
evidence as the extractive generator's verbatim copy-paste would have,
even if the answer was factually correct.

Costs the SAME number of LLM calls as a bare eval.py run of the same
size -- zero extra API calls. The gate-vs-grounding distinction comes
from inspecting result.trace (each pipeline.query() call already returns
this), not from any additional query. If 'generate' never appears in the
trace, the LLM was never called for that question at all.

Run with a real LLM generator (the only way this is actually informative):
    python scripts/diagnose_llm_coverage.py --real-llm gemini --max-test-questions 8
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verityrag.eval import _load, _split  # noqa: E402
from verityrag.pipeline import VerityRAGPipeline  # noqa: E402


def diagnose(embedder=None, generator=None, max_test_questions: int | None = None):
    corpus = _load("corpus.json")
    eval_answerable = _load("eval_answerable.json")
    eval_unanswerable = _load("eval_unanswerable.json")

    calib_pos, test_pos = _split(eval_answerable, calib_fraction=0.5, seed=13)
    calib_neg, _test_neg = _split(eval_unanswerable, calib_fraction=0.5, seed=13)

    if max_test_questions is not None:
        test_pos = test_pos[:max_test_questions]

    pipeline = VerityRAGPipeline(embedder=embedder, generator=generator)
    print(f"Ingesting {len(corpus)} docs...")
    pipeline.ingest_documents(corpus)
    print("Training reranker...")
    pipeline.train_reranker(n_queries=300)
    print("Calibrating sufficiency gate...")
    pipeline.calibrate_sufficiency(
        answerable_questions=[q["question"] for q in calib_pos],
        unanswerable_questions=[q["question"] for q in calib_neg],
    )

    gate_rejected = []
    grounding_rejected = []
    answered_successfully = []

    print(f"\nChecking {len(test_pos)} answerable test questions "
          f"({len(test_pos)} LLM calls at most, one per question, same as a bare eval run)...\n")
    for item in test_pos:
        result = pipeline.query(item["question"])
        trace_actions = [s.action for s in result.trace]
        reached_generation = "generate" in trace_actions

        record = {
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "trace_actions": trace_actions,
            "generated_answer": result.raw_generated_text,
            "grounding_score": round(result.grounding.overall_score, 4) if result.grounding else None,
            "grounding_verdict": result.grounding.verdict if result.grounding else None,
        }

        if not reached_generation:
            gate_rejected.append(record)
        elif result.abstained:
            grounding_rejected.append(record)
        else:
            answered_successfully.append(record)

    print("=" * 70)
    print(f"RESULTS  (generator: {pipeline.generator.name}, embedder: {type(pipeline.embedder).__name__})")
    print("=" * 70)
    print(f"Answered successfully:                    {len(answered_successfully)}/{len(test_pos)}")
    print(f"Gate-rejected (LLM never called):          {len(gate_rejected)}/{len(test_pos)}")
    print(f"Grounding-rejected (LLM called, answer rejected): {len(grounding_rejected)}/{len(test_pos)}")

    if grounding_rejected:
        print("\n--- GROUNDING-REJECTED (the interesting case -- LLM answered, grounding said no) ---")
        for r in grounding_rejected:
            print(f"  Q: {r['question']}")
            print(f"     gold answer:      {r['gold_answer']}")
            print(f"     LLM generated:    {r['generated_answer']}")
            print(f"     grounding score:  {r['grounding_score']}  (verdict: {r['grounding_verdict']})")
            print()
        print("If the LLM's generated answer above looks factually correct despite a low")
        print("grounding score, that supports the paraphrase hypothesis: the grounding")
        print("checker's literal/semantic matching may be under-scoring fluent rephrasing")
        print("that the extractive generator's verbatim copying would never trigger.")

    if gate_rejected:
        print("\n--- GATE-REJECTED (unrelated to the LLM -- same as any generator would see) ---")
        for r in gate_rejected:
            print(f"  Q: {r['question']}")

    return {
        "generator": pipeline.generator.name,
        "embedder": type(pipeline.embedder).__name__,
        "total": len(test_pos),
        "answered_successfully": answered_successfully,
        "gate_rejected": gate_rejected,
        "grounding_rejected": grounding_rejected,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--real-embeddings", action="store_true")
    parser.add_argument("--real-llm", choices=["anthropic", "gemini"], default=None)
    parser.add_argument("--max-test-questions", type=int, default=8)
    args = parser.parse_args()

    chosen_embedder = None
    if args.real_embeddings:
        from verityrag.embedding import SentenceTransformerEmbedder
        chosen_embedder = SentenceTransformerEmbedder()

    chosen_generator = None
    if args.real_llm == "anthropic":
        from verityrag.generation import LLMGenerator
        from verityrag.llm_providers import anthropic_complete_fn
        chosen_generator = LLMGenerator(complete_fn=anthropic_complete_fn)
    elif args.real_llm == "gemini":
        from verityrag.generation import LLMGenerator
        from verityrag.llm_providers import gemini_complete_fn
        chosen_generator = LLMGenerator(complete_fn=gemini_complete_fn)

    result = diagnose(embedder=chosen_embedder, generator=chosen_generator,
                       max_test_questions=args.max_test_questions)

    out_path = Path(__file__).resolve().parents[1] / "llm_coverage_diagnosis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull detail written to {out_path}")