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
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from verityrag import VerityRAGPipeline  # noqa: E402
from verityrag.streaming import LiveDocumentStream  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"

st.set_page_config(page_title="VerityRAG", page_icon="🔎", layout="wide")


# ----------------------------------------------------------------------
# Shared, module-level pipeline -- one instance for every visitor, not
# one per browser session (see module docstring for why).
# ----------------------------------------------------------------------
@st.cache_resource
def _get_shared_pipeline():
    return VerityRAGPipeline()


pipeline = _get_shared_pipeline()
st.session_state.pipeline = pipeline  # same shared object, exposed for introspection/testing


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
    st.caption("Added to the shared index everyone sees, not just this browser.")
    with st.form("add_doc", clear_on_submit=True):
        title = st.text_input("Title")
        text = st.text_area("Text", height=100)
        if st.form_submit_button("Ingest", use_container_width=True) and title and text:
            n = pipeline.ingest_document(f"user-{int(time.time() * 1000)}", title, text)
            st.toast(f"Added {n} chunk(s) — searchable immediately.", icon="✅")
            st.rerun()

    st.divider()
    if st.button("↺ Reset shared demo (clears index for everyone)", use_container_width=True):
        _get_shared_pipeline.clear()  # invalidates the cached singleton -- next access builds a fresh one
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------
tab_ask, tab_demo = st.tabs(["Ask a question", "Real-time streaming demo"])

with tab_ask:
    st.header("Ask a question")
    if pipeline.index.size() == 0:
        st.info("Index is empty — load the demo corpus or add a document from the sidebar first.")

    question = st.text_input(
        "Question", placeholder="e.g. What is Nikola Tesla known for?", key="ask_question"
    )
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and question.strip():
        with st.spinner("Retrieving → reranking → checking groundedness..."):
            result = pipeline.query(question)

        if result.abstained:
            st.warning(
                "**Abstained** — the sufficiency gate didn't find confident, "
                "relevant evidence, so no answer was generated rather than "
                "risk a plausible-sounding guess."
            )
        else:
            st.success(result.answer)
            score = result.grounding.overall_score
            st.progress(
                score,
                text=f"Grounding score: {score:.0%} — verdict: {result.grounding.verdict}",
            )

            with st.expander("Claim-level grounding breakdown"):
                for claim in result.grounding.claims:
                    icon = "✅" if claim.is_grounded else "❌"
                    st.markdown(f"{icon} {claim.claim}")
                    st.caption(
                        f"semantic support: {claim.semantic_support:.2f} · "
                        f"lexical support: {claim.lexical_support:.2f}"
                    )

        with st.expander("Agent trace (hops, reformulation, decisions)"):
            for step in result.trace:
                st.write(
                    f"**hop {step.hop} · {step.action}** — "
                    f"score={step.top_score:.3f} — query used: _{step.query_used!r}_"
                )

        if result.evidence:
            with st.expander(f"Evidence used ({len(result.evidence)} chunks)"):
                for e in result.evidence:
                    st.markdown(f"**{e.title}** — dense={e.dense_score:.3f}, lexical={e.lexical_score:.3f}")
                    st.caption(e.text)

with tab_demo:
    st.header("Real-time streaming demo")
    st.caption(
        "Recreates scripts/demo_streaming.py in the browser: a topic is "
        "refused before it's ingested, streamed in live via a background "
        "thread, then answered immediately after — no restart. "
        "Uses its own dedicated, pre-calibrated index, independent of the "
        "sidebar (the sidebar's full corpus load includes Tesla from the "
        "start, which would defeat the before/after contrast here)."
    )

    if not _data_files_present():
        st.warning("Run `python data/build_corpus.py` first (see sidebar).")
    elif st.session_state.demo_pipeline is None:
        st.info("This sets up a separate index (all topics except Tesla) and "
                "calibrates it — takes ~20-30s, once.")
        if st.button("Set up demo"):
            with st.spinner("Ingesting corpus, training reranker, calibrating..."):
                demo_pipeline = VerityRAGPipeline()
                corpus = _load_json("corpus.json")
                non_tesla_docs = [d for d in corpus if d["title"] != "Nikola_Tesla"]
                demo_pipeline.ingest_documents(non_tesla_docs)
                demo_pipeline.train_reranker(n_queries=200)
                # exclude Tesla questions from calibration too, or calibration
                # would implicitly "see" the topic this demo is about to stream in
                answerable = [q for q in _load_json("eval_answerable.json")
                              if "Tesla" not in q["question"]][:60]
                unanswerable = [q for q in _load_json("eval_unanswerable.json")
                                if "Tesla" not in q["question"]][:60]
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