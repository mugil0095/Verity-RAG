"""
Document chunking for ingestion.

Splits raw document text into overlapping, sentence-aligned chunks so that:
  - chunks stay under a max token budget (approximated by whitespace tokens)
  - each chunk retains enough surrounding context (overlap) for retrieval
  - chunk boundaries fall on sentence edges, not mid-sentence
"""
import re
from dataclasses import dataclass, field

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int  # index of this chunk within its parent document


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    doc_id: str,
    title: str,
    text: str,
    max_tokens: int = 120,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """
    Greedily pack sentences into chunks of at most `max_tokens` whitespace tokens,
    carrying the last `overlap_sentences` sentences of a chunk into the next chunk
    so that context isn't lost at chunk boundaries.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    position = 0

    def flush():
        nonlocal current, current_len, position
        if not current:
            return
        chunk_text = " ".join(current)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::c{position}",
                doc_id=doc_id,
                title=title,
                text=chunk_text,
                position=position,
            )
        )
        position += 1

    for sent in sentences:
        sent_len = len(sent.split())
        if current and current_len + sent_len > max_tokens:
            flush()
            # carry overlap forward
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_len = sum(len(s.split()) for s in current)
        current.append(sent)
        current_len += sent_len

    flush()
    return chunks
