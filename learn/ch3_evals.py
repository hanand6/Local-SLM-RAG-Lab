# ============================================================
# Chapter 3 — Evals
#
# Lab A: build a small golden set → RAG answers it → an LLM judge
#        grades each answer → you grade too → judge/human agreement.
# Lab B: pick a confidence threshold from the business cost of each
#        kind of error (precision/recall trade-off). No model calls.
# ============================================================

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import OPENROUTER_JUDGE_MODEL
from learn.common import call_model, interview_card, model_picker, needs_index

MAX_ROWS = 5

JUDGE_SYSTEM = (
    "You are a strict grader. Compare a candidate answer with a reference answer for the "
    "same question. The candidate is CORRECT if it states the same key facts as the reference "
    "(wording may differ, extra correct detail is fine). It is INCORRECT if a key fact is "
    "missing, wrong, or contradicted, or if it says it cannot answer when the reference does. "
    'Reply with JSON only: {"verdict": "correct" or "incorrect", "reason": "<one sentence>"}'
)


def concept() -> None:
    st.markdown("""
An **eval** is a repeatable test that tells you whether a change made your AI product better or worse. Without evals, every prompt tweak or model swap is a guess.

**Offline vs online**

| | Offline evals | Online evals |
|---|---|---|
| When | Before shipping, on every change | After shipping, on real traffic |
| Data | A fixed **golden set**: questions + expected answers | Real user sessions |
| Signals | Accuracy, faithfulness, refusal rate, latency, cost | Thumbs up/down, re-asks, escalations, task completion, A/B tests |
| Strength | Fast, repeatable, safe | Measures what users actually experience |
| Weakness | Only as good as the golden set | Noisy, slow, and harms are already live |

You need both. Offline catches regressions before launch; online tells you if the offline set still reflects reality.

**Kinds of metrics**
- **Reference-based:** compare to an expected answer (correctness, similarity). Needs a labelled golden set.
- **Reference-free:** judge the answer on its own: is it *faithful* to the retrieved context? Is it *relevant* to the question? Is the retrieved context precise? (These are the RAGAS metrics in the 🧪 Eval tab.)
- **Efficiency:** P50/P95 latency, cost per query, tokens per answer, error and retry rate.

**LLM-as-judge**
A strong model grades answers, which scales far better than human grading. But judges have biases: they prefer longer answers, favour the first option shown, and favour their own model family. So **calibrate the judge against human labels** on a sample and report the agreement rate. If agreement is low, fix the rubric before trusting the scores.

**Thresholds come from business cost**
Many AI features act on a confidence score: auto-answer or hand off, flag or ignore. Where you set the threshold trades **precision** (when we act, how often are we right?) against **recall** (of everything we should act on, how much do we catch?). The right point depends on the **cost of each kind of mistake**, not on maximising accuracy. Lab B makes this concrete.
""")


# ── Lab A: golden set + LLM judge ─────────────────────────────

def _parse_verdict(text: str) -> tuple[str, str]:
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            v = str(data.get("verdict", "")).lower()
            if v in ("correct", "incorrect"):
                return v, str(data.get("reason", ""))
        except json.JSONDecodeError:
            pass
    low = (text or "").lower()
    if "incorrect" in low:
        return "incorrect", text.strip()[:200]
    if "correct" in low:
        return "correct", text.strip()[:200]
    return "unparsed", (text or "").strip()[:200]


def _lab_a(rag) -> None:
    st.markdown("### Lab A: golden set + LLM-as-judge")
    st.markdown(
        f"Write up to {MAX_ROWS} questions about your document, with the answer *you* expect. "
        "The app answers them with RAG, a judge model grades each answer, then you grade too. "
        "The question is how often the judge agrees with you."
    )
    default = pd.DataFrame({"question": [""] * 3, "expected answer": [""] * 3})
    golden = st.data_editor(
        st.session_state.get("ch3_golden", default),
        num_rows="dynamic", use_container_width=True, key="ch3_golden_editor",
        column_config={
            "question": st.column_config.TextColumn("Question", width="large"),
            "expected answer": st.column_config.TextColumn("Expected answer (reference)", width="large"),
        },
    )
    rows = [
        (str(q).strip(), str(a).strip())
        for q, a in zip(golden["question"], golden["expected answer"])
        if str(q).strip() and str(a).strip() and str(q) != "None"
    ][:MAX_ROWS]

    c1, c2 = st.columns(2)
    with c1:
        model = model_picker("Answering model", key="ch3_model", prefer="recursive/flint-v1.0:free")
    with c2:
        judge = model_picker("Judge model", key="ch3_judge", prefer=OPENROUTER_JUDGE_MODEL)
    if model and judge and model == judge:
        st.caption("⚠️ Same model as answerer and judge: self-preference bias. Fine for practice, avoid it in real evals.")
    st.caption(f"This lab makes **{2 * len(rows)} model calls** ({len(rows)} answers + {len(rows)} grades).")

    if st.button("▶️ Answer and grade", type="primary", key="ch3_run",
                 disabled=not (rows and model and judge)):
        st.session_state.ch3_golden = golden
        out = []
        bar = st.progress(0.0)
        for i, (q, ref) in enumerate(rows):
            bar.progress(i / len(rows), text=f"[{i+1}/{len(rows)}] answering...")
            chunks = rag.retrieve(q, top_k=5)
            system, prompt = rag.build_prompt(q, chunks)
            ans = call_model(rag, model, system, prompt)
            answer = ans.response or f"(error: {ans.error})"

            bar.progress((i + 0.5) / len(rows), text=f"[{i+1}/{len(rows)}] grading...")
            g = call_model(
                rag, judge, JUDGE_SYSTEM,
                f"Question: {q}\n\nReference answer: {ref}\n\nCandidate answer: {answer}",
            )
            verdict, reason = _parse_verdict(g.response) if not g.error else ("error", g.error)
            out.append({
                "question": q, "expected": ref, "answer": answer,
                "judge": verdict, "judge reason": reason, "you": None,
                "latency s": round(ans.wall_clock_s, 1), "tokens": ans.prompt_tokens + ans.output_tokens,
            })
        bar.empty()
        st.session_state.ch3_results = pd.DataFrame(out)

    res = st.session_state.get("ch3_results")
    if res is None or res.empty:
        return

    st.markdown("**Grade each answer yourself** in the *you* column, then compare with the judge.")
    graded = st.data_editor(
        res, use_container_width=True, hide_index=True, key="ch3_grade_editor",
        disabled=[c for c in res.columns if c != "you"],
        column_config={
            "you": st.column_config.SelectboxColumn("you", options=["correct", "incorrect"]),
            "answer": st.column_config.TextColumn(width="large"),
            "judge reason": st.column_config.TextColumn(width="medium"),
        },
    )

    judged = graded[graded["judge"].isin(["correct", "incorrect"])]
    both = judged[judged["you"].isin(["correct", "incorrect"])]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Judge accuracy score", f"{(judged['judge'] == 'correct').mean():.0%}" if len(judged) else "—")
    m2.metric("Your accuracy score", f"{(both['you'] == 'correct').mean():.0%}" if len(both) else "—")
    m3.metric("Judge ↔ you agreement", f"{(both['judge'] == both['you']).mean():.0%}" if len(both) else "—",
              help="Of the answers you graded, how often the judge gave the same verdict.")
    m4.metric("P50 latency", f"{graded['latency s'].median():.1f}s")
    if len(both) and (both["judge"] != both["you"]).any():
        st.warning(
            "The judge disagreed with you on some answers. Read its reasons: is the rubric unclear, "
            "or is the judge biased (e.g. rewarding longer answers)? In a real eval you'd fix the "
            "rubric and re-measure before trusting judge scores."
        )
    st.caption(
        "With 3–5 rows these numbers are anecdotes, not statistics. Real golden sets have 100+ "
        "questions, drawn from real user queries and covering the hard cases."
    )


# ── Lab B: threshold from business cost ───────────────────────

PRESETS = {
    "Customer support bot": (4.0, 12.0,
        "Handing off costs ~$4 of agent time; a wrong auto-answer costs ~$12 (follow-up contact, some churn risk)."),
    "Medical / legal advice": (5.0, 500.0,
        "A wrong answer can cause real harm. Handing off to a professional is cheap by comparison."),
    "Movie trivia app": (1.0, 0.5,
        "A wrong answer is mildly annoying; not answering loses the user. Cheap mistakes."),
}


@st.cache_data
def _simulated_queries(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Confidence scores for n queries; correct answers tend to score higher."""
    rng = np.random.default_rng(seed)
    correct = rng.random(n) < 0.7
    score = np.where(correct, rng.beta(4, 2, n), rng.beta(2, 3, n))
    return pd.DataFrame({"score": score, "correct": correct})


def _metrics(df: pd.DataFrame, t: float, c_handoff: float, c_wrong: float) -> dict:
    act = df["score"] >= t
    tp = int((act & df["correct"]).sum())       # auto-answered, right
    fp = int((act & ~df["correct"]).sum())      # auto-answered, wrong
    fn = int((~act & df["correct"]).sum())      # handed off, would have been right
    tn = int((~act & ~df["correct"]).sum())     # handed off, would have been wrong
    handoffs = fn + tn
    return {
        "t": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "coverage": (tp + fp) / len(df),
        "cost": fp * c_wrong + handoffs * c_handoff,
    }


def _lab_b() -> None:
    st.markdown("### Lab B: choose a threshold from the cost of mistakes")
    st.markdown(
        "A RAG assistant gets **400 questions**. For each it has a confidence score (e.g. retrieval "
        "similarity). Above the threshold it **auto-answers**; below it **hands off to a human**. "
        "These are simulated scores, so no model calls."
    )
    preset = st.radio("Scenario", list(PRESETS), horizontal=True, key="ch3_preset")
    p_handoff, p_wrong, why = PRESETS[preset]
    st.caption(why.replace("$", "\\$"))   # unescaped $…$ renders as LaTeX
    c1, c2, c3 = st.columns(3)
    with c1:
        c_handoff = st.number_input("Cost of a hand-off ($)", 0.0, 1000.0, p_handoff, key=f"ch3_ch_{preset}")
    with c2:
        c_wrong = st.number_input("Cost of a wrong auto-answer ($)", 0.0, 10000.0, p_wrong, key=f"ch3_cw_{preset}")
    with c3:
        t = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01, key="ch3_t")

    df = _simulated_queries()
    sweep = [_metrics(df, x, c_handoff, c_wrong) for x in np.linspace(0, 1, 101)]
    best = min(sweep, key=lambda m: m["cost"])
    cur = _metrics(df, t, c_handoff, c_wrong)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precision", f"{cur['precision']:.0%}", help="Of auto-answers, share that were right.")
    k2.metric("Recall", f"{cur['recall']:.0%}", help="Of answerable questions, share we auto-answered.")
    k3.metric("Auto-answered", f"{cur['coverage']:.0%}")
    k4.metric("Total cost", f"${cur['cost']:,.0f}",
              delta=f"${cur['cost'] - best['cost']:,.0f} vs best", delta_color="inverse")

    fig = go.Figure()
    xs = [m["t"] for m in sweep]
    fig.add_trace(go.Scatter(x=xs, y=[m["cost"] for m in sweep], name="Total cost ($)", yaxis="y2",
                             line=dict(color="#ffb347", width=3)))
    fig.add_trace(go.Scatter(x=xs, y=[m["precision"] for m in sweep], name="Precision", line=dict(color="#3effa3")))
    fig.add_trace(go.Scatter(x=xs, y=[m["recall"] for m in sweep], name="Recall", line=dict(color="#6C8EF5")))
    fig.add_vline(x=t, line_dash="dash", annotation_text="your threshold")
    fig.add_vline(x=best["t"], line_dash="dot", line_color="#ffb347", annotation_text="lowest cost",
                  annotation_position="bottom right")
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Threshold",
        yaxis=dict(title="Precision / recall", range=[0, 1.05]),
        yaxis2=dict(title="Cost ($)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"| | Would be right | Would be wrong |\n|---|---|---|\n"
        f"| **Auto-answered** | ✅ {cur['tp']} | ❌ {cur['fp']} (cost \\${c_wrong:g} each) |\n"
        f"| **Handed off** | ↩️ {cur['fn']} | ↩️ {cur['tn']} |\n\n"
        f"Hand-offs cost \\${c_handoff:g} each. **Lowest-cost threshold: {best['t']:.2f}** "
        f"(precision {best['precision']:.0%}, recall {best['recall']:.0%}, cost \\${best['cost']:,.0f})."
    )
    st.markdown("""
**What to look for**
- Switch scenarios. The *same model* gets a very different best threshold when mistakes cost more.
- In **Medical**, the best point has high precision and low recall: answer rarely, but be right.
- In **Trivia**, it drops to zero: always answering beats ever handing off.
- **Support** sits in between, balancing precision and recall.
- In an interview, say it this way: *"I'd set the threshold where the expected cost of errors is lowest, using what each kind of mistake costs the business."*
""")


def lab(rag) -> None:
    tab_a, tab_b = st.tabs(["A · Golden set + judge", "B · Threshold & cost"])
    with tab_a:
        if not needs_index(rag):
            _lab_a(rag)
    with tab_b:
        _lab_b()


def interview() -> None:
    interview_card(
        question="How would you design an AI evaluation system for resumes?",
        asked_by="Microsoft",
        structure=[
            "Clarify the task: screening? ranking? extracting skills? Who acts on the output?",
            "Define success with stakeholders: what a 'good' output is, and which errors are costly.",
            "Offline: a golden set of labelled resumes covering roles, formats and demographic slices.",
            "Metrics: precision/recall at the decision threshold, extraction accuracy, fairness across groups.",
            "LLM-as-judge for scale, calibrated against recruiter labels; humans review edge cases.",
            "Online: recruiter overrides, time-to-hire, adverse-impact monitoring; staged rollout.",
        ],
        sample=(
            "I'd start by pinning down the decision. Say we're shortlisting resumes for recruiters. "
            "The costly error is rejecting a qualified candidate, so I'd tune for high recall and let "
            "recruiters handle precision. Offline, I'd build a golden set of about 500 resumes labelled "
            "by experienced recruiters, stratified by role, seniority, resume format and demographic "
            "group. I'd measure recall and precision at our threshold, and check the selection rate "
            "for each group for adverse impact. An LLM judge can score rationale quality at scale, but "
            "only after it agrees with recruiters on a calibration sample. Online, I'd track recruiter "
            "override rates and downstream interview pass rates, and roll out in stages."
        ),
        change_my_mind=(
            "If the system only extracted fields (skills, dates) rather than making decisions, I'd "
            "focus on extraction accuracy and drop most of the fairness tooling, because the risk "
            "profile is much lower."
        ),
        pitfalls=[
            "Ignoring fairness and legal risk: hiring is high-stakes and regulated.",
            "Using one accuracy number instead of cost-aware precision and recall.",
            "Trusting an LLM judge without calibrating it against humans.",
            "No online monitoring after launch.",
        ],
    )
    st.write("")
    interview_card(
        question="How do you measure whether an LLM project is efficient? And how do you measure model quality?",
        asked_by="NVIDIA",
        structure=[
            "Split the two: efficiency (cost, speed) vs quality (is the output good).",
            "Efficiency: P50/P95 latency, time to first token, cost per query or task, tokens per answer, throughput, error and retry rate.",
            "Quality: task metrics on a golden set (accuracy, faithfulness, refusal rate) plus online signals.",
            "Tie them together: quality per dollar, and a minimum quality bar before optimising cost.",
            "Name the levers: smaller or routed models, caching, shorter prompts, batching.",
        ],
        sample=(
            "I treat them as two scorecards with one rule: never trade quality below a set bar for "
            "efficiency. For quality, I use a golden set with task metrics (answer correctness, and "
            "faithfulness for RAG) plus online signals like thumbs-down rate and escalations. For "
            "efficiency, I track P95 latency (because the tail is what users feel), cost per successful "
            "task rather than per call, and tokens per answer. Then I compare options on quality per "
            "dollar. A small model that clears the quality bar at a fifth of the cost wins, and routing "
            "only hard queries to the big model often captures most of the gain."
        ),
        change_my_mind=(
            "For a low-volume, high-stakes use case (like legal review), I'd mostly ignore cost and "
            "optimise for quality and auditability."
        ),
        pitfalls=[
            "Reporting average latency instead of P95.",
            "Cost per call instead of cost per successful task: retries and failures hide there.",
            "Optimising efficiency without a quality floor.",
        ],
    )


CHAPTER = {
    "id": "ch3",
    "title": "Evals",
    "short": "Evals",
    "tagline": "How you know a change made things better: golden sets, judges, and cost-aware thresholds.",
    "concept": concept,
    "lab": lab,
    "interview": interview,
}
