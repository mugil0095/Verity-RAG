"""
Builds a realistic (not toy/synthetic) working corpus and evaluation set from the
real SQuAD 1.1 dev set (Wikipedia paragraphs + human-written questions).

Downloads the SQuAD dev set itself if it isn't already present next to this
script -- no manual download step required. All paths are resolved relative
to this script's own location (not the current working directory), so this
works whether you run it as `python data/build_corpus.py` from the repo
root, `python build_corpus.py` from inside data/, or with a full/absolute
path, on any OS.

Output (written next to this script, i.e. into data/):
  corpus.json            -> list of {doc_id, title, text}  (documents to ingest)
  eval_answerable.json   -> real questions answerable from corpus, with gold answers
  eval_unanswerable.json -> real questions whose topic is EXCLUDED from the corpus,
                             used to test the agent's abstention / hallucination guard
"""
import json
import random
import urllib.request
from pathlib import Path

random.seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent
SQUAD_PATH = SCRIPT_DIR / "squad_dev.json"
SQUAD_URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json"

if not SQUAD_PATH.exists():
    print(f"squad_dev.json not found -- downloading from {SQUAD_URL} ...")
    urllib.request.urlretrieve(SQUAD_URL, SQUAD_PATH)
    print(f"Saved to {SQUAD_PATH}")

with open(SQUAD_PATH) as f:
    squad = json.load(f)

topics = squad["data"]

# Use the first N topics as the "ingested corpus" (answerable domain)
N_INCLUDED = 12
included_topics = topics[:N_INCLUDED]
excluded_topics = topics[N_INCLUDED:]  # ALL remaining topics held out -> unanswerable-question pool

corpus = []
eval_answerable = []
doc_id = 0

for topic in included_topics:
    title = topic["title"]
    for para in topic["paragraphs"]:
        text = para["context"]
        corpus.append({"doc_id": f"d{doc_id}", "title": title, "text": text})
        # take at most 2 real questions per paragraph to keep eval set manageable
        for qa in para["qas"][:2]:
            if qa.get("is_impossible"):
                continue
            if not qa["answers"]:
                continue
            eval_answerable.append({
                "question": qa["question"],
                "gold_answer": qa["answers"][0]["text"],
                "source_doc_id": f"d{doc_id}",
                "source_title": title,
            })
        doc_id += 1

# Unanswerable set: real questions from topics we never ingested
eval_unanswerable = []
for topic in excluded_topics:
    title = topic["title"]
    for para in topic["paragraphs"][:6]:
        for qa in para["qas"][:2]:
            if qa.get("is_impossible"):
                continue
            eval_unanswerable.append({
                "question": qa["question"],
                "excluded_topic": title,
            })

random.shuffle(eval_answerable)
random.shuffle(eval_unanswerable)
eval_answerable = eval_answerable[:150]
eval_unanswerable = eval_unanswerable[:150]

with open(SCRIPT_DIR / "corpus.json", "w") as f:
    json.dump(corpus, f)
with open(SCRIPT_DIR / "eval_answerable.json", "w") as f:
    json.dump(eval_answerable, f, indent=2)
with open(SCRIPT_DIR / "eval_unanswerable.json", "w") as f:
    json.dump(eval_unanswerable, f, indent=2)

print(f"corpus docs: {len(corpus)}")
print(f"answerable eval questions: {len(eval_answerable)}")
print(f"unanswerable(held-out) eval questions: {len(eval_unanswerable)}")
print(f"included topics: {[t['title'] for t in included_topics]}")
print(f"excluded topics used for abstention test: {[t['title'] for t in excluded_topics]}")
