# ============================================================
# Chapter 2 — Hallucination and grounding
#
# Lab: strict grounded prompt vs loose prompt, with
#   - an abstention gate on retrieval similarity (refuse before calling the model)
#   - a citation check (do cited chunks exist in what was retrieved?)
#   - a crude support check (which answer sentences overlap the context?)
# ============================================================

from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from learn.common import answer_box, call_model, interview_card, model_picker, needs_index

SYSTEM_LOOSE = (
    "You are a helpful assistant. Use the context below if it helps, "
    "otherwise answer from your own knowledge. Answer confidently and concisely."
)

_STOP = set("""a an the and or but if of to in on at for with by from as is are was were be been being
this that these those it its it's they them their there here what which who whom whose when where why how
not no yes do does did done can could should would will shall may might must have has had having i you he
she we our your his her my me us than then so such also into about over under more most less least very
just only any all each other some""".split())


def concept() -> None:
    st.markdown("""
A **hallucination** is a fluent, confident answer that isn't supported by facts. Language models are trained to produce *plausible next text*, not *true* text. When they don't know something, the plausible continuation is still an answer.

**Separate the causes. Interviewers listen for this split.**

| Training-time causes | Inference-time causes |
|---|---|
| Gaps or errors in the training data | Retrieval returned the wrong chunks, or none |
| Knowledge cutoff: the world changed since training | The prompt never allows "I don't know" |
| Training rewards answering over admitting uncertainty | The question is ambiguous |
| Rare facts are learned weakly | High temperature (sampling randomness) |
| | Very long context: facts in the middle get missed |

**Mitigations, from cheapest to most involved**
1. **Grounding.** Answer only from retrieved sources, and say so in the system prompt.
2. **Permission to abstain.** Give the model an explicit "not found" answer it can use.
3. **Abstention threshold.** If retrieval similarity is low, don't ask the model at all. Return "I don't know" or hand off to a human.
4. **Citations.** Make the model point to its sources, then *check* the citations exist.
5. **Lower temperature** for factual tasks.
6. **Evals.** Measure faithfulness on a test set so you know if a change helped (Chapter 3).
7. **UX.** Show sources and confidence, and make reporting a wrong answer easy.
8. **Training-time fixes** (model provider's job): better data, and training that rewards calibrated "I don't know".

Grounding reduces hallucination. It doesn't eliminate it: models can still misread or over-generalise the right context.
""")


def _cited_chunks(answer: str) -> list[int]:
    """Chunk numbers the answer cites, e.g. 'chunk 3' → 3."""
    return [int(n) for n in re.findall(r"chunk\W{0,3}(\d+)", answer, flags=re.I)]


def _support_table(answer: str, context: str) -> pd.DataFrame:
    """
    Crude lexical check: for each answer sentence, what share of its content
    words appear in the retrieved context? Low share = possibly unsupported.
    (Real systems use an LLM judge or an NLI model. See Chapter 3.)
    """
    ctx_words = set(re.findall(r"[a-z0-9]+", context.lower()))
    rows = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", answer):
        words = [w for w in re.findall(r"[a-z0-9]+", sent.lower()) if w not in _STOP and len(w) > 2]
        if len(words) < 3:
            continue
        share = sum(w in ctx_words for w in words) / len(words)
        rows.append({
            "Answer sentence": sent.strip()[:160],
            "Words found in context": f"{share:.0%}",
            "Verdict": "✅ supported" if share >= 0.6 else ("🟡 partly" if share >= 0.35 else "🔴 likely unsupported"),
        })
    return pd.DataFrame(rows)


def lab(rag) -> None:
    if needs_index(rag):
        return

    st.markdown(
        "Compare a **strict grounded prompt** with a **loose one** that lets the model use its "
        "own knowledge. Then check the citations and how well each answer is supported."
    )
    model = model_picker("Model", key="ch2_model", prefer="recursive/flint-v1.0:free")

    examples = {
        "Answerable": "What is the main recommendation in the document?",
        "Unanswerable (tempting)": "What was the company's revenue in 1995, and who was CEO then?",
        "Unanswerable (off-topic)": "What is the boiling point of mercury?",
    }
    ex = st.radio("Try a question type", list(examples), horizontal=True, key="ch2_ex")
    question = st.text_input("Question", value=examples[ex], key=f"ch2_q_{ex}")

    c1, c2 = st.columns(2)
    with c1:
        top_k = st.slider("Chunks retrieved (top-k)", 2, 10, 5, key="ch2_k")
    with c2:
        threshold = st.slider(
            "Abstain if best chunk similarity is below", 0.0, 1.0, 0.35, 0.05, key="ch2_thr",
            help="all-MiniLM cosine similarity. Relevant chunks usually score 0.4–0.7, unrelated ones below 0.25.",
        )
    st.caption("This lab makes **1–2 model calls**. The strict run is skipped if it abstains.")

    if st.button("▶️ Run", type="primary", key="ch2_run", disabled=not (model and question.strip())):
        chunks = rag.retrieve(question, top_k=top_k)
        best = max((c["similarity"] for c in chunks), default=0.0)
        context = "\n\n".join(c["text"] for c in chunks)

        abstained = best < threshold
        strict = None
        if not abstained:
            with st.spinner("Strict grounded prompt..."):
                system, prompt = rag.build_prompt(question, chunks)
                strict = call_model(rag, model, system, prompt)
        with st.spinner("Loose prompt..."):
            _, prompt = rag.build_prompt(question, chunks)
            loose = call_model(rag, model, SYSTEM_LOOSE, prompt)
        st.session_state.ch2_results = dict(
            q=question, chunks=chunks, best=best, thr=threshold,
            abstained=abstained, strict=strict, loose=loose, context=context,
        )

    r = st.session_state.get("ch2_results")
    if not r:
        return

    # ── Retrieval confidence ──
    fig = go.Figure(go.Bar(
        x=[f"#{i+1} · chunk {c['chunk_idx']+1}" for i, c in enumerate(r["chunks"])],
        y=[c["similarity"] for c in r["chunks"]],
        marker_color=["#3effa3" if c["similarity"] >= r["thr"] else "#ff6b6b" for c in r["chunks"]],
    ))
    fig.add_hline(y=r["thr"], line_dash="dash", annotation_text=f"abstain threshold {r['thr']:.2f}")
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10),
                      title="Retrieval similarity of each chunk", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    col_s, col_l = st.columns(2)
    with col_s:
        if r["abstained"]:
            st.markdown("**Strict + abstention gate**")
            st.markdown(
                f'<div class="answer-box">🛑 I don\'t know. The best match ({r["best"]:.2f}) is below the '
                f'threshold ({r["thr"]:.2f}), so the model was never asked. '
                "In a product, this hands off to a human or asks the user to rephrase.</div>",
                unsafe_allow_html=True,
            )
        else:
            answer_box("Strict grounded prompt", r["strict"], "Answer only from context, may say 'not found'.")
    with col_l:
        answer_box("Loose prompt", r["loose"], "Context optional, may use own knowledge.")

    # ── Checks ──
    st.markdown("#### 🔍 Checks")
    retrieved = {c["chunk_idx"] + 1 for c in r["chunks"]}
    for label, res in [("Strict", r["strict"]), ("Loose", r["loose"])]:
        if res is None or res.error or not res.response:
            continue
        cited = _cited_chunks(res.response)
        bad = sorted({n for n in cited if n not in retrieved})
        if not cited:
            cite_msg = "⚪ no citations"
        elif bad:
            cite_msg = f"🔴 cites chunk(s) {bad} that were **not** retrieved: invented citations"
        else:
            cite_msg = f"✅ cites chunk(s) {sorted(set(cited))}, all of them retrieved"
        with st.expander(f"{label}: citations {cite_msg.split(' ')[0]} · sentence support", expanded=label == "Loose"):
            st.markdown(f"**Citations:** {cite_msg}")
            table = _support_table(res.response, r["context"])
            if table.empty:
                st.caption("Answer too short to check sentence by sentence.")
            else:
                st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "The support check is a crude word-overlap heuristic, there to make the idea visible. "
        "Production systems use an LLM judge or a natural-language-inference model "
        "(RAGAS 'faithfulness' in the 🧪 Eval tab does this)."
    )

    st.markdown("""
**What to look for**
- On an **unanswerable** question, does the loose prompt answer anyway? That's hallucination invited by the prompt.
- Move the **threshold** up and down. Too high and you refuse answerable questions; too low and weak context gets through. Chapter 3 shows how to pick it from the cost of each mistake.
- Do citations point to chunks that were actually retrieved?
""")


def interview() -> None:
    interview_card(
        question="Users complain Gemini is confident but wrong. How would you fix this?",
        asked_by="Google",
        structure=[
            "Clarify: which queries, how often, how harmful? Get examples and a baseline rate.",
            "Diagnose: separate training-time causes from inference-time causes.",
            "Fix inference-time issues first: grounding, permission to abstain, thresholds, citations.",
            "Make confidence visible in the UX: sources, hedged language when unsure, easy reporting.",
            "Measure: build a hallucination eval set, track faithfulness and abstention rate offline and online.",
            "Longer term: feed failures into training data and calibration work with the model team.",
        ],
        sample=(
            "First I'd size the problem: sample flagged conversations, label which answers were wrong, "
            "and find where it clusters. Fresh facts, niche topics and numbers are common culprits. "
            "Training-time causes like knowledge cutoff need the model team. The fastest wins are at "
            "inference time. I'd ground factual queries in search results, allow and reward 'I don't "
            "know', and add an abstention threshold so low-confidence answers fall back to showing "
            "sources instead of asserting. I'd require citations and check they're real. Success is a "
            "lower hallucination rate on a held-out eval set without a big rise in unhelpful refusals, "
            "plus fewer 'wrong answer' reports online."
        ),
        change_my_mind=(
            "If the errors cluster on reasoning (wrong maths, wrong logic) rather than missing facts, "
            "grounding won't help much. I'd look at tool use (a calculator or code execution) and "
            "model-level fixes instead."
        ),
        pitfalls=[
            "Jumping to \"retrain the model\" without diagnosing first.",
            "Ignoring the refusal trade-off: abstaining too often is also a failure.",
            "No metric: you can't claim a fix without a before/after eval.",
            "Treating citations as proof without checking them.",
        ],
    )


CHAPTER = {
    "id": "ch2",
    "title": "Hallucination and grounding",
    "short": "Grounding",
    "tagline": "Why models are confidently wrong, and the levers that reduce it.",
    "concept": concept,
    "lab": lab,
    "interview": interview,
}
