import json
from pathlib import Path

import pytest

from verityrag.planner import MAX_PARTS, Planner

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
planner = Planner()


def subs(question):
    return planner.plan(question).sub_questions


def test_splits_separate_sentences():
    plan = planner.plan("When was Tesla born? Also, where did Edison die?")
    assert plan.rule == "sentences"
    assert plan.sub_questions == ["When was Tesla born?", "Where did Edison die?"]


def test_splits_one_sentence_on_a_conjunction():
    plan = planner.plan("When was Tesla born, and where did Edison die?")
    assert plan.rule == "conjunction"
    assert plan.sub_questions == ["When was Tesla born?", "Where did Edison die?"]


def test_splits_preposition_led_questions_and_semicolons():
    assert subs("In what year was Warsaw founded and in which city was Chopin born?") == [
        "In what year was Warsaw founded?", "In which city was Chopin born?"]
    assert subs("Who wrote Hamlet; who painted the Mona Lisa?") == [
        "Who wrote Hamlet?", "Who painted the Mona Lisa?"]


def test_splits_into_three_and_accepts_a_missing_final_question_mark():
    assert len(subs("Who wrote Hamlet, and who painted the Mona Lisa, and when was Tesla born?")) == 3
    assert len(subs("Where was Tesla's property sent? Also, in what expression can one find DTIME(n)")) == 2


def test_accepts_a_wh_word_in_the_middle_of_the_first_part():
    parts = subs("Sky Movies also include what optional soundtracks, and who founded Sky?")
    assert parts == ["Sky Movies also include what optional soundtracks?", "Who founded Sky?"]


def test_pronoun_is_fine_when_the_part_names_its_own_subject():
    # "his" refers to Edison, who is named inside the same part.
    assert len(subs("When was Tesla born, and where did Edison keep his notebooks?")) == 2


@pytest.mark.parametrize("question", [
    "What is Nikola Tesla known for?",
    "What are the pros and cons of solar power?",                    # "and" inside a noun phrase
    "What is the capital of France and who is its president?",       # "its" points back at part one
    "Who was the first king and what did he rule?",                  # "he" points back
    "What was the cause and what happened next?",                    # "next" needs the first part
    "Tell me about Tesla",
    "",
])
def test_leaves_single_or_dependent_questions_whole(question):
    plan = planner.plan(question)
    assert plan.sub_questions == [question]
    assert plan.rule == "single"


def test_leaves_questions_with_more_than_max_parts_whole():
    question = " ".join(f"Who wrote book number {i}?" for i in range(MAX_PARTS + 1))
    assert planner.plan(question).sub_questions == [question]


@pytest.mark.skipif(not (DATA_DIR / "eval_answerable.json").exists(), reason="eval data not present")
def test_no_single_part_eval_question_is_split():
    """The 300 eval questions are all single SQuAD questions. If the planner split
    any, the stock eval numbers could change; this is what makes multi-agent mode
    safe to leave on by default."""
    questions = []
    for name in ("eval_answerable.json", "eval_unanswerable.json"):
        with open(DATA_DIR / name) as f:
            questions += [row["question"] for row in json.load(f)]
    assert questions
    assert [q for q in questions if len(subs(q)) > 1] == []
