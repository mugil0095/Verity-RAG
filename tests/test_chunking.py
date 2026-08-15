from verityrag.chunking import chunk_document, split_sentences


def test_split_sentences_basic():
    text = "Tesla was born in 1856. He worked on alternating current. He died in 1943."
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("Tesla")


def test_empty_document_returns_no_chunks():
    assert chunk_document("d1", "T", "") == []
    assert chunk_document("d1", "T", "   ") == []


def test_chunk_respects_max_token_budget():
    long_text = " ".join([f"Sentence number {i} contains several words for length." for i in range(40)])
    chunks = chunk_document("d1", "Title", long_text, max_tokens=50, overlap_sentences=1)
    assert len(chunks) > 1
    for c in chunks:
        # allow a little slack since we don't split mid-sentence
        assert len(c.text.split()) <= 50 + 12


def test_chunk_overlap_carries_context():
    long_text = " ".join([f"Fact {i} is a unique statement about topic {i}." for i in range(30)])
    chunks = chunk_document("d1", "Title", long_text, max_tokens=40, overlap_sentences=1)
    assert len(chunks) >= 2
    # last sentence of chunk N should reappear as first sentence of chunk N+1
    last_sentence_of_first = split_sentences(chunks[0].text)[-1]
    first_sentence_of_second = split_sentences(chunks[1].text)[0]
    assert last_sentence_of_first == first_sentence_of_second


def test_chunk_ids_are_unique_and_ordered():
    text = " ".join([f"Point {i} states something new here today." for i in range(20)])
    chunks = chunk_document("docX", "T", text, max_tokens=30)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.position for c in chunks] == list(range(len(chunks)))
