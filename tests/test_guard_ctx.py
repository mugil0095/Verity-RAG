from verityrag.guard.ctx import select_context


def test_short_context_unchanged():
    assert select_context("One short sentence.", "anything") == "One short sentence."


def test_keeps_answer_relevant_sentence_within_budget():
    ctx = " ".join([f"Filler sentence number {i} about nothing." for i in range(60)] + ["The tower is 330 metres tall."])
    out = select_context(ctx, "The tower is 330 metres tall", max_words=30)
    assert "330 metres" in out and len(out.split()) <= 40
