"""Verity-RAG as an MCP server: any MCP client/agent can ingest documents, ask grounded questions and verify claims.
Run:  python -m verityrag.mcp_server   (stdio transport)"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .pipeline import VerityRAGPipeline

mcp = FastMCP("verity-rag")
_pipe = VerityRAGPipeline()
_guard = None


def _get_guard():
    global _guard
    if _guard is None:
        from .guard import LearnedGuard
        _guard = LearnedGuard()
    return _guard


@mcp.tool()
def ingest(doc_id: str, title: str, text: str) -> int:
    """Add a document to the live index. Returns the number of chunks indexed."""
    return _pipe.ingest_document(doc_id, title, text)


@mcp.tool()
def answer(question: str) -> dict:
    """Answer a question from ingested documents, or abstain when evidence is insufficient."""
    r = _pipe.query(question)
    out = {"answer": r.answer, "abstained": r.abstained, "hops": r.hops_used,
           "sources": [{"chunk_id": c.chunk_id, "text": c.text[:300]} for c in r.evidence[:3]]}
    if r.answer is not None:
        try:
            g = _get_guard().score(r.answer, r.evidence, question)
            out["guard"] = {"p_hallucinated": g.p_hallucinated, "verdict": g.verdict}
        except Exception as e:  # guard model optional
            out["guard"] = {"error": str(e)}
    return out


@mcp.tool()
def verify_claim(claim: str, context: str, question: str = "") -> dict:
    """Check whether a claim/answer is supported by the given context (learned hallucination detector)."""
    g = _get_guard().score(claim, context, question)
    return {"p_hallucinated": g.p_hallucinated, "verdict": g.verdict, "latency_ms": g.latency_ms}


if __name__ == "__main__":
    mcp.run()
