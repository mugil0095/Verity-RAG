"""
Streamlit frontend for VerityRAG.

Uses the real VerityRAGPipeline directly in-process (the same class that
powers api.py) -- no separate server to run, just:

    streamlit run app.py

The main pipeline is a SHARED, module-level singleton (like api.py's own
pipeline), not per-session state -- deliberately changed from an earlier
per-session design. Each browser session used to get its own independent
in-memory index, which is a genuinely nicer experience (one visitor's
"add a document" or "reset" never affects anyone else), but it means the
~422MB vector matrix (845 chunks x 65536-dim HashingEmbedder vectors)
gets duplicated in memory for every concurrent visitor. On a memory-
constrained free host (~1GB), that's the difference between supporting
roughly 1 concurrent fully-loaded session and several dozen. For a
public demo meant to be shown, not a private sandbox per visitor, that
trade-off is worth it -- loading the corpus, resetting, and adding a
document now affect the shared state for everyone currently using the
app, and the UI says so explicitly rather than changing this silently.

The SEPARATE demo_pipeline (the "Real-time streaming demo" tab) stays
per-session on purpose: it specifically needs each visitor to see a
fresh "abstained before streaming" state for the before/after contrast
to mean anything. Sharing it would mean only the very first visitor to
ever click "Run the live streaming demo" sees a real demonstration --
everyone after that would find Tesla already ingested.
"""
import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from verityrag import VerityRAGPipeline  # noqa: E402
from verityrag.streaming import LiveDocumentStream  # noqa: E402
from verityrag.usage_tracker import log_input  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"

st.set_page_config(page_title="VerityRAG", page_icon="🔎", layout="wide")

# gemini_complete_fn (llm_providers.py) reads GEMINI_API_KEY from
# os.environ -- it predates this app and is also used from the CLI,
# where that's the natural place to look. Streamlit's own secrets
# mechanism (st.secrets) is separate from os.environ by default, so this
# bridges the two rather than modifying the already-tested library
# function to know about Streamlit specifically. st.secrets.get() itself
# raises StreamlitSecretNotFoundError (not just returning None) when no
# secrets.toml exists at all -- confirmed directly, not assumed -- so
# this needs the same try/except usage_tracker.py's _get_secret() already
# uses for exactly this reason. Silently does nothing if no secret is
# configured -- same "optional feature, not core to the app working"
# principle as usage_tracker.py's GITHUB_TOKEN handling.
try:
    _gemini_secret = st.secrets.get("GEMINI_API_KEY")
except Exception:
    _gemini_secret = None
if _gemini_secret and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = _gemini_secret


# ----------------------------------------------------------------------
# Shared, module-level pipeline -- one instance for every visitor, not
# one per browser session (see module docstring for why).
# ----------------------------------------------------------------------
@st.cache_resource
def _get_shared_pipeline():
    return VerityRAGPipeline()


pipeline = _get_shared_pipeline()
st.session_state.pipeline = pipeline  # same shared object, exposed for introspection/testing


def _llm_comparison_available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _build_llm_comparison_agent():
    """Shares the main pipeline's expensive, already-loaded resources
    (index, embedder, reranker, sufficiency gate) -- only the generator
    differs. Deliberately NOT cached: the reranker and sufficiency gate
    on the shared pipeline.agent can change after this app starts (a
    visitor training the reranker or calibrating later), and caching a
    comparison agent built before that would hold a stale reference to
    whatever those were at cache time. Constructing an AgentController
    itself is cheap -- it only stores references, no heavy computation --
    so building it fresh on every call is the simpler, correct choice."""
    from verityrag.agent import AgentController
    from verityrag.generation import LLMGenerator
    from verityrag.llm_providers import gemini_complete_fn

    return AgentController(
        index=pipeline.index,
        embedder=pipeline.embedder,
        generator=LLMGenerator(complete_fn=gemini_complete_fn),
        reranker_model=pipeline.agent.reranker_model,
        sufficiency_gate=pipeline.agent.sufficiency_gate,
        max_hops=pipeline.agent.max_hops,
    )


# ----------------------------------------------------------------------
# Session state -- only for things that genuinely should stay per-visitor
# (the separate streaming-demo pipeline), plus UI flags that mirror the
# shared pipeline's own state (trained/calibrated are properties of
# `pipeline` itself, tracked here only so Streamlit's rerun model has
# something to read without re-deriving it every time).
# ----------------------------------------------------------------------
def _init_state():
    if "demo_pipeline" not in st.session_state:
        st.session_state.demo_pipeline = None  # separate, self-contained, per-session on purpose -- see tab_demo
        st.session_state.demo_before = None
        st.session_state.demo_after = None
        st.session_state.demo_unanswerable = None


_init_state()


def _data_files_present() -> bool:
    return all((DATA_DIR / f).exists() for f in
               ("corpus.json", "eval_answerable.json", "eval_unanswerable.json"))


def _load_json(name):
    import json
    with open(DATA_DIR / name) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Sidebar: corpus setup + stats
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 VerityRAG")
    st.caption("Real-time agentic RAG with grounding & hallucination detection")
    st.caption("Questions asked and documents added here may be logged for demo/analytics purposes.")

    st.divider()
    st.subheader("Index status")
    st.caption("Shared across all visitors — not private per browser session.")
    c1, c2 = st.columns(2)
    c1.metric("Chunks indexed", pipeline.index.size())
    c2.metric("Live updates", pipeline.index.updates_count)
    reranker_trained = pipeline._reranker_model is not None
    gate_calibrated = pipeline.agent.sufficiency_gate.is_calibrated
    st.write(
        f"Reranker: {'✅ trained' if reranker_trained else '⬜ not trained'}  \n"
        f"Sufficiency gate: {'✅ calibrated' if gate_calibrated else '⬜ default (uncalibrated)'}"
    )

    st.divider()
    st.subheader("Load the real demo corpus")
    st.caption("620 real Wikipedia paragraphs (SQuAD dev set) + trains the reranker. "
               "Loads it for everyone currently using the app, not just you.")
    if not _data_files_present():
        st.warning("data/corpus.json not found.")
        st.code("python data/build_corpus.py", language="bash")
    else:
        if st.button("Load corpus + train reranker", use_container_width=True):
            with st.spinner("Ingesting 620 documents and training the reranker..."):
                corpus = _load_json("corpus.json")
                pipeline.ingest_documents(corpus)
                pipeline.train_reranker(n_queries=200)
            st.toast(f"Indexed {pipeline.index.size()} chunks.", icon="✅")
            st.rerun()

        if pipeline.index.size() > 0 and not gate_calibrated:
            if st.button("Calibrate sufficiency gate (~30s, recommended)", use_container_width=True):
                with st.spinner("Calibrating on real labeled questions..."):
                    answerable = _load_json("eval_answerable.json")[:60]
                    unanswerable = _load_json("eval_unanswerable.json")[:60]
                    ok = pipeline.calibrate_sufficiency(
                        [q["question"] for q in answerable],
                        [q["question"] for q in unanswerable],
                    )
                if ok:
                    st.toast("Calibrated — the hallucination guard is now meaningfully stronger.", icon="✅")
                else:
                    st.toast("Calibration needs a bigger index first.", icon="⚠️")
                st.rerun()

    st.divider()
    st.subheader("Add your own document")
    st.caption("Added to the shared index everyone sees, not just this browser. "
               "Limited to 200 / 5,000 chars — public, shared memory, kept bounded.")
    with st.form("add_doc", clear_on_submit=True):
        title = st.text_input("Title", max_chars=200)
        text = st.text_area("Text", height=100, max_chars=5000)
        if st.form_submit_button("Ingest", use_container_width=True) and title and text:
            n = pipeline.ingest_document(f"user-{int(time.time() * 1000)}", title, text)
            log_input("document", {"title": title, "text": text, "chunks_added": n})
            st.toast(f"Added {n} chunk(s) — searchable immediately.", icon="✅")
            st.rerun()

    st.divider()
    confirm_reset = st.checkbox("I understand this clears the index for every current visitor, not just me")
    if st.button("↺ Reset shared demo", use_container_width=True, disabled=not confirm_reset):
        _get_shared_pipeline.clear()  # invalidates the cached singleton -- next access builds a fresh one
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------
tab_ask, tab_demo = st.tabs(["Ask a question", "Real-time streaming demo"])


def _render_result(result, container=st):
    """Renders one AgentResult's abstain/answer/grounding/claims/trace/
    evidence -- factored out so the single-result and side-by-side LLM
    comparison cases share the exact same display logic instead of
    diverging into two separately-maintained code paths."""
    if result.abstained:
        container.warning(
            "**Abstained** — the sufficiency gate didn't find confident, "
            "relevant evidence, so no answer was generated rather than "
            "risk a plausible-sounding guess."
        )
    else:
        container.success(result.answer)
        score = result.grounding.overall_score
        container.progress(
            score,
            text=f"Grounding score: {score:.0%} — verdict: {result.grounding.verdict}",
        )

        with container.expander("Claim-level grounding breakdown"):
            for claim in result.grounding.claims:
                icon = "✅" if claim.is_grounded else "❌"
                st.markdown(f"{icon} {claim.claim}")
                st.caption(
                    f"semantic support: {claim.semantic_support:.2f} · "
                    f"lexical support: {claim.lexical_support:.2f}"
                )

    with container.expander("Agent trace (hops, reformulation, decisions)"):
        for step in result.trace:
            st.write(
                f"**hop {step.hop} · {step.action}** — "
                f"score={step.top_score:.3f} — query used: _{step.query_used!r}_"
            )

    if result.evidence:
        with container.expander(f"Evidence used ({len(result.evidence)} chunks)"):
            for e in result.evidence:
                st.markdown(f"**{e.title}** — dense={e.dense_score:.3f}, lexical={e.lexical_score:.3f}")
                st.caption(e.text)


with tab_ask:
    st.header("Ask a question")
    if pipeline.index.size() == 0:
        st.info("Index is empty — load the demo corpus or add a document from the sidebar first.")

    question = st.text_input(
        "Question", placeholder="e.g. What is Nikola Tesla known for?", key="ask_question", max_chars=500
    )

    compare_with_llm = st.checkbox(
        "Also show a real LLM's answer (Gemini) side-by-side — same retrieved evidence, "
        "different generator, showcases this project's actual grounding/hallucination thesis",
        disabled=not _llm_comparison_available(),
        help=None if _llm_comparison_available() else
             "Needs a GEMINI_API_KEY configured in this app's secrets to enable.",
    )
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and question.strip():
        with st.spinner("Retrieving → reranking → checking groundedness..."):
            result = pipeline.query(question)

        log_payload = {
            "question": question,
            "abstained": result.abstained,
            "answer": result.answer,
            "grounding_score": result.grounding.overall_score if result.grounding else None,
        }

        if compare_with_llm:
            col_extractive, col_llm = st.columns(2)
            with col_extractive:
                st.subheader("Extractive (default)")
                _render_result(result, container=col_extractive)

            with col_llm:
                st.subheader("Real LLM (Gemini)")
                try:
                    with st.spinner("Calling Gemini..."):
                        llm_agent = _build_llm_comparison_agent()
                        llm_result = llm_agent.answer(question)
                    _render_result(llm_result, container=col_llm)
                    log_payload["llm_comparison"] = {
                        "abstained": llm_result.abstained,
                        "answer": llm_result.answer,
                        "grounding_score": llm_result.grounding.overall_score if llm_result.grounding else None,
                    }
                except Exception as e:
                    # Real, expected failure modes here: a free-tier daily
                    # quota exhausted (llm_providers.py raises a clear
                    # RuntimeError for this specifically), a transient
                    # network issue, or any other real API failure. This
                    # comparison is an optional extra, not core to the
                    # app -- a visitor should still get their real,
                    # working extractive answer on the left even if the
                    # LLM side fails for any reason, not a crashed page.
                    col_llm.error(f"LLM comparison unavailable right now: {e}")
                    log_payload["llm_comparison_error"] = str(e)

        else:
            _render_result(result)

        log_input("question", log_payload)

with tab_demo:
    st.header("Real-time streaming demo")
    st.caption(
        "Recreates scripts/demo_streaming.py in the browser: a topic is "
        "refused before it's ingested, streamed in live via a background "
        "thread, then answered immediately after — no restart. "
        "Uses its own dedicated, pre-calibrated index (a 9-topic subset, "
        "not the full corpus — kept smaller since, unlike the sidebar's "
        "shared index, this one is per-visitor), independent of the "
        "sidebar (whose full corpus load includes Tesla from the start, "
        "which would defeat the before/after contrast here)."
    )

    if not _data_files_present():
        st.warning("Run `python data/build_corpus.py` first (see sidebar).")
    elif st.session_state.demo_pipeline is None:
        st.info("This sets up a separate, per-visitor index (a small topic "
                 "subset, not the full corpus — see caption above) and "
                 "calibrates it — takes ~10-15s, once.")
        if st.button("Set up demo"):
            with st.spinner("Ingesting corpus, training reranker, calibrating..."):
                demo_pipeline = VerityRAGPipeline()
                corpus = _load_json("corpus.json")
                # A small, fixed topic subset, not "everything except Tesla" --
                # this pipeline is per-session by necessity (each visitor needs
                # a fresh "abstained before streaming" state), so unlike the
                # shared main pipeline, its memory cost is paid by EVERY
                # visitor who tries this tab, not once. The full non-Tesla
                # corpus is 528 docs, nearly as large as the shared pipeline's
                # own 620 -- on a ~1GB host, even 1-2 concurrent visitors
                # trying this tab could exceed the budget. The demo's actual
                # requirement is just "some genuine topic diversity, none of
                # it Tesla" -- these two smallest topics (47 docs total) are
                # more than enough for that.
                demo_topics = {"Sky_(United_Kingdom)", "Victoria_(Australia)", "Southern_California",
                               "Huguenot", "Normans", "Steam_engine", "Computational_complexity_theory",
                               "Warsaw", "Super_Bowl_50"}
                demo_docs = [d for d in corpus if d["title"] in demo_topics]
                demo_pipeline.ingest_documents(demo_docs)
                demo_pipeline.train_reranker(n_queries=200)
                # Filter ANSWERABLE calibration questions to the same topic
                # subset, via source_title -- these are tied to a specific
                # ingested topic, so this matters now that most of the 12
                # topics aren't ingested for this demo. UNANSWERABLE questions
                # need no such filtering: they're drawn from topics never in
                # the corpus at all (a different field, excluded_topic, on
                # entirely different source data) -- inherently unrelated to
                # any of the 12 topics regardless of which subset is ingested.
                answerable = [q for q in _load_json("eval_answerable.json")
                              if q["source_title"] in demo_topics][:60]
                unanswerable = _load_json("eval_unanswerable.json")[:60]
                demo_pipeline.calibrate_sufficiency(
                    [q["question"] for q in answerable],
                    [q["question"] for q in unanswerable],
                )
                st.session_state.demo_pipeline = demo_pipeline
            st.rerun()
    else:
        demo_pipeline = st.session_state.demo_pipeline
        demo_question = "What did Nikola Tesla contribute to electrical engineering?"
        st.write(f'Demo question: *"{demo_question}"*')

        if st.session_state.demo_before is None:
            if st.button("▶ Run the live streaming demo"):
                corpus = _load_json("corpus.json")
                tesla_docs = [d for d in corpus if d["title"] == "Nikola_Tesla"][:6]

                st.session_state.demo_before = demo_pipeline.query(demo_question)

                stream = LiveDocumentStream(demo_pipeline, delay_seconds=0.05)
                stream.start()
                stream.enqueue_many(tesla_docs)
                progress = st.progress(0, text="Starting stream...")
                total = len(tesla_docs)
                while stream.stats.docs_ingested < total:
                    pct = stream.stats.docs_ingested / total
                    progress.progress(
                        pct, text=f"Streaming live: {stream.stats.docs_ingested}/{total} documents ingested..."
                    )
                    time.sleep(0.1)
                stream.stop()
                progress.progress(1.0, text=f"Done — {stream.stats.docs_ingested} documents streamed in.")

                st.session_state.demo_after = demo_pipeline.query(demo_question)
                st.session_state.demo_unanswerable = demo_pipeline.query("What caused the 1973 oil crisis?")
                st.rerun()
        else:
            col_before, col_after = st.columns(2)
            with col_before:
                st.subheader("Before streaming")
                r = st.session_state.demo_before
                st.error("Abstained" if r.abstained else r.answer)
            with col_after:
                st.subheader("After streaming (same question)")
                r = st.session_state.demo_after
                if r.abstained:
                    st.error("Abstained")
                else:
                    st.success(r.answer)
                    st.caption(f"grounding: {r.grounding.overall_score:.0%} ({r.grounding.verdict})")

            st.divider()
            st.subheader('Genuinely out-of-domain question: "What caused the 1973 oil crisis?"')
            r = st.session_state.demo_unanswerable
            if r.abstained:
                st.success("✅ Correctly abstained — this topic was never ingested, so no answer was fabricated.")
            else:
                st.error(f"⚠️ Answered anyway: {r.answer}")

            if st.button("↺ Reset this demo"):
                st.session_state.demo_pipeline = None
                st.session_state.demo_before = None
                st.session_state.demo_after = None
                st.session_state.demo_unanswerable = None
                st.rerun()