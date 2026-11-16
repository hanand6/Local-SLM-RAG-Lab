# ============================================================
# learn/ui.py — the "🎓 Learn" tab
#
# Each chapter = Concept (short read) → Lab (hands-on, uses your
# indexed document) → Interview card (practise a committed answer).
# Progress lives in the Streamlit session (cleared on restart).
# ============================================================

from __future__ import annotations

import streamlit as st

from learn import (
    ch1_prompt_rag_ft, ch2_grounding, ch3_evals,
    ch4_latency_cost, ch5_agents, ch6_system_design,
    ch7_ml_literacy, ch8_launch_rai,
)
from config import PERSIST_INDEX
from learn import progress
from learn.common import calls_used

# Learning order (not the order topics appear in interview prep lists):
# grounding before evals, because the eval labs reuse grounded answers;
# ML literacy before system design, which pulls everything together;
# launches last, because shipping is the final step.
CHAPTERS = [
    ch1_prompt_rag_ft.CHAPTER,
    ch2_grounding.CHAPTER,
    ch3_evals.CHAPTER,
    ch4_latency_cost.CHAPTER,
    ch5_agents.CHAPTER,
    ch7_ml_literacy.CHAPTER,
    ch6_system_design.CHAPTER,
    ch8_launch_rai.CHAPTER,
]

COMING_SOON: list[str] = []


def render_learn(rag) -> None:
    progress.load_once()          # restore saved progress + drafts (first render)
    done: set = st.session_state.setdefault("learn_done", set())

    st.header("🎓 Learn")
    st.caption(
        "Short chapters on the AI topics that come up in PM interviews. "
        "Each has a concept, a lab that runs on your own document, and an interview card."
    )

    nav, body = st.columns([1, 3], gap="large")

    with nav:
        st.markdown("**Chapters**")
        # Static labels: changing a radio's options would reset the selection.
        labels = [f"{i}. {c['title']}" for i, c in enumerate(CHAPTERS, 1)]
        choice = st.radio("Chapter", labels, key="learn_chapter", label_visibility="collapsed")
        chapter = CHAPTERS[labels.index(choice)]

        st.progress(len(done) / len(CHAPTERS), text=f"{len(done)}/{len(CHAPTERS)} complete")
        st.markdown(
            "  \n".join(
                f"{'✅' if c['id'] in done else '⬜'} {i}. {c['short']}"
                for i, c in enumerate(CHAPTERS, 1)
            )
        )
        st.caption(f"🔢 Model calls made in labs this session: **{calls_used()}**")
        st.caption("Free models allow roughly 20 calls/min. Each lab shows its call count before you run it.")

        with st.expander("Saved progress"):
            st.caption(
                "Completed chapters and interview drafts are saved to "
                "`data/learn_progress.json` and survive restarts. Lab results are not saved."
                if PERSIST_INDEX else "Saving is off (PERSIST_INDEX=false): progress resets on restart."
            )
            if st.checkbox("I want to start over", key="learn_reset_confirm"):
                # on_click runs before the next render, so it may reset widget state.
                st.button("Reset progress and drafts", key="learn_reset", on_click=progress.reset)

        if COMING_SOON:
            with st.expander("Coming later"):
                st.markdown("\n".join(f"- {t}" for t in COMING_SOON))

    with body:
        st.subheader(chapter["title"])
        st.caption(chapter["tagline"])

        t_concept, t_lab, t_interview = st.tabs(["📖 Concept", "🧪 Lab", "🎤 Interview"])
        with t_concept:
            chapter["concept"]()
        with t_lab:
            chapter["lab"](rag)
        with t_interview:
            chapter["interview"]()

        st.divider()
        is_done = chapter["id"] in done
        if st.checkbox("Mark this chapter complete", value=is_done, key=f"done_{chapter['id']}"):
            if not is_done:
                done.add(chapter["id"])
                progress.save()
                st.rerun()
        elif is_done:
            done.discard(chapter["id"])
            progress.save()
            st.rerun()
