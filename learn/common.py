# ============================================================
# learn/common.py — shared helpers for the Learn chapters
#
#   call_model()        route a prompt to any configured model and count calls
#   model_picker()      selectbox limited to the models ticked in the sidebar
#   interview_card()    consistent layout for interview practice
#   answer_box()        render a model answer with its key numbers
# ============================================================

from __future__ import annotations

import html
import zlib

import streamlit as st

from benchmarker import BenchmarkResult
from config import MODELS
from learn import progress


# ── Model calls ──────────────────────────────────────────────

def call_model(rag, model: str, system: str, prompt: str, on_token=None) -> BenchmarkResult:
    """
    Run one prompt on any backend (Ollama or cloud) and count the call.
    on_token(text_so_far) streams the answer where the backend supports it.
    """
    backend = MODELS.get(model, {}).get("backend", "ollama")
    if backend == "ollama":
        res = rag.benchmarker.run_single(model=model, prompt=prompt, system=system)
    else:
        res = rag._run_cloud_model(model=model, system=system, prompt=prompt, on_token=on_token)
    st.session_state.learn_calls = st.session_state.get("learn_calls", 0) + 1
    return res


def calls_used() -> int:
    return st.session_state.get("learn_calls", 0)


def model_picker(label: str, key: str, prefer: str | None = None) -> str | None:
    """Pick from the models ticked in the sidebar."""
    options = st.session_state.get("selected_models") or []
    if not options:
        st.warning("Tick at least one model in the sidebar.")
        return None
    index = options.index(prefer) if prefer in options else 0
    return st.selectbox(
        label, options, index=index, key=key,
        format_func=lambda m: MODELS.get(m, {}).get("label", m),
    )


def needs_index(rag) -> bool:
    """Show a hint and return True when no document is indexed yet."""
    if rag.chunk_count == 0:
        st.info(
            "This lab uses your own document. Index one in **📄 Upload & Index** first, "
            "then come back.",
            icon="📄",
        )
        return True
    return False


# ── Rendering ────────────────────────────────────────────────

def md_safe(text: str) -> str:
    """Escape $ so Streamlit markdown doesn't render '$4 ... $12' as LaTeX."""
    return (text or "").replace("$", "\\$")


def html_safe(text: str) -> str:
    """For text placed inside unsafe_allow_html blocks: escape HTML and $."""
    return html.escape(text or "").replace("$", "&#36;").replace("\n", "<br>")


def answer_box(title: str, res: BenchmarkResult | None, note: str = "") -> None:
    st.markdown(f"**{title}**")
    if note:
        st.caption(note)
    if res is None:
        st.markdown('<div class="answer-box">—</div>', unsafe_allow_html=True)
        return
    text = res.response or (f"❌ {res.error}" if res.error else "⚠️ No response.")
    st.markdown(
        f'<div class="answer-box">{html_safe(text)}</div>',
        unsafe_allow_html=True,
    )
    if not res.error:
        st.caption(
            f"🧾 {res.prompt_tokens:,} prompt tokens · {res.output_tokens:,} output"
            + (f" ({res.reasoning_tokens:,} thinking)" if res.reasoning_tokens else "")
            + f" · ⏱️ {res.wall_clock_s:.1f}s"
        )


def interview_card(question: str, asked_by: str, structure: list[str], sample: str,
                   change_my_mind: str, pitfalls: list[str]) -> None:
    """Interview practice: question → try it yourself → reveal a model answer."""
    with st.container(border=True):
        st.markdown(f"#### 🎤 “{question}”")
        st.caption(f"Typical of: {asked_by}")
        key = f"draft_{zlib.crc32(question.encode())}"
        st.text_area(
            "Your answer (draft it before revealing — say it out loud in 2 minutes)",
            value=progress.drafts().get(key, ""),
            key=progress.widget_key(key), height=110,
            on_change=progress.on_draft_change, args=(key,),
            help="Saved automatically when you click outside the box.",
        )
        with st.expander("Reveal: a strong answer"):
            st.markdown("**Structure**")
            st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(structure, 1)))
            st.markdown("**Sample answer (committed, not hedged)**")
            st.markdown(f"> {sample}")
            st.markdown(f"**What would change my mind:** {change_my_mind}")
            st.markdown("**Common pitfalls**")
            st.markdown("\n".join(f"- {p}" for p in pitfalls))
