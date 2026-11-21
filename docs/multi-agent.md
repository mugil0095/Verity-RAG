Two-part questions ("When was Tesla born? Also, where did Edison die?") broke the single loop. On 74 of them it answered 72 and put both gold answers in the text for 11 (15%). Retrieval ran once for the whole question, so the top chunks usually covered one half (both answers were in the evidence for 30 of 74), and the grounding check let the half-answers through because it only scores whether each sentence is supported by the evidence.

`multi_agent.py` adds a planner and runs a retriever and a verifier per sub-question. The planner (`planner.py`) splits only when every part is a complete question on its own: separate sentences, or one sentence joined by "and" or ";" before a wh-word. A pronoun that points back at the first part ("who is its president") or a fragment ("what happened next") keeps the question whole. None of the 300 single-part eval questions gets split, so they take the old path and the stock numbers (coverage 93.3%, guard 77.3%) are unchanged. The planner costs about 8 microseconds per question. Retriever and verifier are the old loop cut in two, `AgentController.retrieve` and `AgentController.respond`, so each sub-question gets its own retrieval, sufficiency gate, grounding check and optional learned guard. A part that fails its checks is named in the answer ("Not answered (insufficient evidence): ...") and listed in `parts` on the `/query` response, next to the plan, the abstain reason and every agent's steps.

Results on 74 two-part questions (`python -m verityrag.compound_eval`, 95% Wilson intervals):

| mode | both answers present | at least one |
|---|---|---|
| single-agent | 11/74 (15%, 9-25%) | 53/74 (72%) |
| multi-agent | 30/74 (41%, 30-52%) | 64/74 (86%) |
| perfect split | 31/74 (42%, 31-53%) | 63/74 (85%) |

The planner split 67 of the 74 pairs. 19 questions flipped to correct and none flipped to wrong (exact McNemar p = 4e-06). The perfect-split row asks each half on its own and joins the answers. No planner on this retriever and generator can beat its 42%. The extractive generator puts the gold answer in only 47 of 75 single questions (63%), and 0.63 squared is about 0.4. The other 58% is lost in retrieval and generation. A split question runs one retrieval per part, so p50 latency goes from 45 ms to 99 ms for two parts in the sandbox, while single questions stay at 45 ms.

Free-text phrasing is untested. The two join templates are the surface patterns the planner parses, so the table shows decomposition at its best. The 74 pairs come from 75 test questions, each used in two pairs, so they overlap. Scoring is the keyword proxy from `eval.py`. An LLM planner would cover dependent and free-form phrasing, and `Planner` is the one class to swap. `pipeline.multi_agent = False` restores the old loop for every question.
