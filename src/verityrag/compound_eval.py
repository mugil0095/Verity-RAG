"""
Two-part-question eval: does the planner -> retriever -> verifier split help
when one question is really two?

Every eval question in data/ is a single SQuAD question, so the split can't
move those numbers (the planner leaves all 300 whole; see tests/test_planner.py).
This eval builds questions it CAN win or lose on: pairs of answerable questions
joined by one of two templates,

    "<A?> Also, <b?>"          (two sentences)
    "<A>, and <b?>"            (one sentence, conjunction)

Pairs come from the TEST half of the answerable questions only (the sufficiency
gate was calibrated on the other half; same split and seed as eval.py), and
the same pairs are asked in three modes:

    single-agent   the original loop (pipeline.multi_agent = False)
    multi-agent    planner -> retriever -> verifier
    perfect split  each half asked on its own and the answers joined: the
                   ceiling for ANY planner with this retriever and generator

Scoring is eval.py's keyword proxy: a gold answer counts when its text appears
in the answer. The joining templates are the two surface patterns the
rule-based planner handles, so the multi-agent row measures "does decomposition
help", not "can rules parse arbitrary phrasing" (see planner.py).

Run with: python -m verityrag.compound_eval
"""
from __future__ import annotations

import os

# Same OpenMP guard as eval.py, before anything imports LightGBM.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402

from .eval import _load, _split  # noqa: E402
from .pipeline import VerityRAGPipeline  # noqa: E402
from .planner import Planner  # noqa: E402


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:]


def _has(gold: str, text: str | None) -> bool:
    return gold.lower() in (text or "").lower()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / d, (centre + half) / d


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def build_pairs(test_pos: list[dict], seeds: tuple[int, ...] = (101, 202)) -> list[dict]:
    """Deterministic pairs of test questions, each question used at most once per
    shuffle. Returns [{"question", "template", "a", "b"}]."""
    seen, pairs = set(), []
    for seed in seeds:
        items = list(test_pos)
        random.Random(seed).shuffle(items)
        for a, b in zip(items[0::2], items[1::2]):
            key = frozenset([a["question"], b["question"]])
            if key in seen:
                continue
            seen.add(key)
            template = "sentences" if len(pairs) % 2 == 0 else "conjunction"
            qa, qb = a["question"], b["question"]
            question = (f"{qa} Also, {_lower_first(qb)}" if template == "sentences"
                        else f"{qa.rstrip('?').rstrip()}, and {_lower_first(qb)}")
            pairs.append({"question": question, "template": template, "a": a, "b": b})
    return pairs


def build_pipeline() -> tuple[VerityRAGPipeline, list[dict]]:
    """Same setup as eval.run_eval; returns the pipeline and the TEST answerable rows."""
    corpus = _load("corpus.json")
    calib_pos, test_pos = _split(_load("eval_answerable.json"), 0.5, 13)
    calib_neg, _ = _split(_load("eval_unanswerable.json"), 0.5, 13)
    pipeline = VerityRAGPipeline()
    pipeline.ingest_documents(corpus)
    pipeline.train_reranker(n_queries=300)
    pipeline.calibrate_sufficiency([q["question"] for q in calib_pos], [q["question"] for q in calib_neg])
    return pipeline, test_pos


def _score(result, a: dict, b: dict) -> dict:
    ans = (not result.abstained) and result.answer
    return {
        "answered": not result.abstained,
        "a": bool(ans and _has(a["gold_answer"], result.answer)),
        "b": bool(ans and _has(b["gold_answer"], result.answer)),
        "ev": any(_has(a["gold_answer"], c.text) for c in result.evidence)
        and any(_has(b["gold_answer"], c.text) for c in result.evidence),
    }


def evaluate_pairs(pipeline: VerityRAGPipeline, pairs: list[dict]) -> dict:
    planner = Planner()
    modes = {"single_agent": [], "multi_agent": [], "perfect_split": []}
    split = 0
    for p in pairs:
        a, b, q = p["a"], p["b"], p["question"]
        split += len(planner.plan(q).sub_questions) > 1

        pipeline.multi_agent = False
        modes["single_agent"].append(_score(pipeline.query(q), a, b))
        pipeline.multi_agent = True
        modes["multi_agent"].append(_score(pipeline.query(q), a, b))

        ra, rb = pipeline.agent.answer(a["question"]), pipeline.agent.answer(b["question"])
        modes["perfect_split"].append({
            "answered": (not ra.abstained) and (not rb.abstained),
            "a": (not ra.abstained) and _has(a["gold_answer"], ra.answer),
            "b": (not rb.abstained) and _has(b["gold_answer"], rb.answer),
            "ev": any(_has(a["gold_answer"], c.text) for c in ra.evidence)
            and any(_has(b["gold_answer"], c.text) for c in rb.evidence),
        })

    def summarize(rows: list[dict], idx: list[int] | None = None) -> dict:
        rows = rows if idx is None else [rows[i] for i in idx]
        return {
            "answered": sum(r["answered"] for r in rows),
            "at_least_one_answer": sum(r["a"] or r["b"] for r in rows),
            "both_answers": sum(r["a"] and r["b"] for r in rows),
            "both_in_evidence": sum(r["ev"] for r in rows),
        }

    both = {m: [r["a"] and r["b"] for r in rows] for m, rows in modes.items()}
    multi_only = sum(m and not s for m, s in zip(both["multi_agent"], both["single_agent"]))
    single_only = sum(s and not m for m, s in zip(both["multi_agent"], both["single_agent"]))
    by_template = {
        t: {m: summarize(rows, [i for i, p in enumerate(pairs) if p["template"] == t])["both_answers"]
            for m, rows in modes.items()}
        for t in ("sentences", "conjunction")
    }
    return {
        "pairs": len(pairs),
        "split_by_planner": split,
        **{m: summarize(rows) for m, rows in modes.items()},
        "mcnemar_both_answers": {
            "multi_only": multi_only, "single_only": single_only,
            "p_value": round(mcnemar_exact(multi_only, single_only), 6),
        },
        "both_answers_by_template": by_template,
    }


def _fmt(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({k / n:.0%}, CI {lo:.0%}-{hi:.0%})"


def print_report(report: dict) -> None:
    n = report["pairs"]
    print(f"Two-part questions: {n} pairs from the test half; planner split {report['split_by_planner']}")
    print(f"{'mode':<14}{'answered':<22}{'>=1 answer':<22}{'BOTH answers':<24}both in evidence")
    for mode in ("single_agent", "multi_agent", "perfect_split"):
        r = report[mode]
        print(f"{mode.replace('_', '-'):<14}{_fmt(r['answered'], n):<22}{_fmt(r['at_least_one_answer'], n):<22}"
              f"{_fmt(r['both_answers'], n):<24}{_fmt(r['both_in_evidence'], n)}")
    m = report["mcnemar_both_answers"]
    print(f"Both answers, multi vs single: multi-only {m['multi_only']}, single-only {m['single_only']}, "
          f"exact McNemar p = {m['p_value']}")


if __name__ == "__main__":
    pipeline, test_pos = build_pipeline()
    result = evaluate_pairs(pipeline, build_pairs(test_pos))
    print_report(result)
    print(json.dumps(result, indent=2))
