"""Evidence selection: keep the context sentences most relevant to the answer so a small cross-encoder
sees the part that decides support (also the production path: retrieved chunks are already short)."""
import re

_W = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to", "for", "and", "or", "as", "by", "with", "at", "from", "that", "this", "it", "its", "be", "has", "have", "had"}


def select_context(context: str, answer: str, max_words: int = 120) -> str:
    if len(context.split()) <= max_words:
        return context
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", context) if s.strip()]
    ans = {w for w in _W.findall(answer.lower()) if w not in _STOP}
    score = [len(ans & set(_W.findall(s.lower()))) / (1 + 0.02 * len(s.split())) for s in sents]
    keep, used = set(), 0
    for i in sorted(range(len(sents)), key=lambda i: -score[i]):
        n = len(sents[i].split())
        if used + n > max_words and keep:
            continue
        keep.add(i); used += n
        if used >= max_words:
            break
    return " ".join(sents[i] for i in sorted(keep))
