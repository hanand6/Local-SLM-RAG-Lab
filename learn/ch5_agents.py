# ============================================================
# Chapter 5 — Agents and safeguards
#
# Lab A: run a small tool-using agent on your document, with an
#        approval policy, a recipient allowlist and a step budget.
# Lab B: error compounding — why long agent chains fail.
# Lab C: drill — when NOT to use an agent.
# ============================================================

from __future__ import annotations

import html
import json

import plotly.graph_objects as go
import streamlit as st

from learn import agent as ag
from learn.common import call_model, html_safe, interview_card, md_safe, model_picker, needs_index

SAMPLE_TASKS = {
    "Multi-step + approval": (
        "Find the main risk in the document. Calculate what a 9% increase on 120,000,000 would be. "
        "Then email a two-sentence summary to cfo@acme.com."
    ),
    "Blocked by policy": "Email a summary of the document's key numbers to reporter@newsdaily.com.",
    "Agent overkill": "What is 4,200,000,000 divided by 12?",
    "Reversible action": "Create a high-priority ticket to follow up on the biggest risk in the document.",
}

STATUS_ICON = {"auto": "⚙️", "approved": "✅", "rejected": "🚫", "blocked": "⛔", "error": "⚠️", "final": "🏁"}


def concept() -> None:
    st.markdown("""
An **agent** is an LLM in a loop: it decides on an action, calls a **tool** (search, calculator, email, an API), reads the result, and decides again until the task is done. A chatbot *says* things; an agent *does* things.

**Why agents are risky: errors compound.** If each step is right 95% of the time, a 10-step task succeeds only 0.95¹⁰ ≈ **60%** of the time. And unlike a wrong chat answer, a wrong *action* (an email sent, money moved, a record deleted) may not be reversible.

**Safeguards, in layers**

| Layer | Example |
|---|---|
| **Classify actions by risk** | Read-only (search) → reversible (create a draft or ticket) → irreversible (send, pay, delete) |
| **Human in the loop** | Pause for approval before irreversible actions. Autonomy grows with trust and evals |
| **Hard policy** | Rules no model output can override: allowlisted recipients, spending caps, no production deletes |
| **Least privilege** | Give the agent only the tools and permissions the task needs |
| **Budgets** | Max steps, max spend, timeouts. Stop runaway loops |
| **Treat tool output as data** | Text from web pages, emails or documents can contain injected instructions (*prompt injection*) |
| **Audit trail** | Log every thought, action and approval so failures can be traced |
| **Evals** | Task success rate, wrong-action rate and escalation rate, measured before widening autonomy |

**When *not* to use an agent**
- The steps are known in advance → write a fixed workflow (deterministic, cheaper, testable).
- A single call answers it → just call the model or a calculator.
- Errors are costly and hard to detect, with no good place for human review.
- Latency matters: every loop step is another model call.

Rule of thumb: **use the least autonomy that solves the problem.** Workflow before agent; human approval before full autonomy.
""")


# ── Lab A: the agent ──────────────────────────────────────────

def _render_trace(run: ag.AgentRun) -> None:
    for i, s in enumerate(run.steps, 1):
        icon = STATUS_ICON.get(s["status"], "•")
        if s["tool"] is None:
            st.markdown(f"{icon} **Step {i}: final answer**. _{md_safe(html.escape(s['thought']))}_")
            continue
        risk = ag.TOOLS.get(s["tool"], {}).get("risk", "?")
        with st.expander(f"{icon} Step {i}: `{s['tool']}` [{risk}] · {s['status']}", expanded=s["status"] in ("blocked", "rejected")):
            st.markdown(f"**Thought:** {md_safe(html.escape(s['thought']))}")
            st.code(json.dumps(s["args"], indent=2, ensure_ascii=False), language="json")
            st.markdown("**Observation:**")
            st.text(s["observation"][:1500])
            if s.get("raw"):
                st.caption(f"Raw reply: {s['raw']}")


def _lab_agent(rag) -> None:
    st.markdown(
        "A small agent with four tools. **search_docs** and **calculator** are read-only, "
        "**create_ticket** is reversible, and **send_email** is irreversible (it writes to a mock "
        "outbox, so nothing is really sent). Every step is shown, so you can see where agents go wrong."
    )
    model = model_picker("Agent model", key="ch5_model", prefer="nvidia/nemotron-3.5-lightning:free")
    pick = st.radio("Sample task", list(SAMPLE_TASKS), horizontal=True, key="ch5_pick")
    task = st.text_area("Task", SAMPLE_TASKS[pick], key=f"ch5_task_{pick}", height=80)
    c1, c2 = st.columns([3, 2])
    with c1:
        policy = st.radio("Approval policy", list(ag.POLICIES), format_func=ag.POLICIES.get, key="ch5_policy")
    with c2:
        domains = st.text_input("Allowed email domains (hard policy)", "acme.com", key="ch5_domains")
    st.caption(f"Up to **{ag.MAX_STEPS + 1} model calls** (one per step). Free models allow about 20/min.")

    if st.button("▶️ Start agent", type="primary", key="ch5_run", disabled=not (model and task.strip())):
        run = ag.AgentRun(
            task=task.strip(), model=model, policy=policy,
            allowed_domains=tuple(d.strip().lower() for d in domains.split(",") if d.strip()),
        )
        st.session_state.ch5_agent_run = run
        _advance(rag, run)

    run: ag.AgentRun | None = st.session_state.get("ch5_agent_run")
    if run is None:
        return

    st.markdown(f"**Task:** {md_safe(html.escape(run.task))}")
    _render_trace(run)

    if run.status == "awaiting_approval" and run.pending:
        p = run.pending
        with st.container(border=True):
            st.markdown(f"### ✋ Approval needed: `{p['tool']}` [{ag.TOOLS[p['tool']]['risk']}]")
            st.markdown(f"**Agent's reason:** {md_safe(html.escape(p['thought']))}")
            st.code(json.dumps(p["args"], indent=2, ensure_ascii=False), language="json")
            note = st.text_input("Note to the agent (optional, sent if you reject)", key="ch5_note")
            a, r, _ = st.columns([1, 1, 3])
            if a.button("✅ Approve", type="primary", key="ch5_approve"):
                ag.resolve(run, rag, approved=True)
                _advance(rag, run)
                st.rerun()
            if r.button("🚫 Reject", key="ch5_reject"):
                ag.resolve(run, rag, approved=False, note=note)
                _advance(rag, run)
                st.rerun()

    if run.status in ("done", "failed"):
        (st.success if run.status == "done" else st.error)(
            ("🏁 " if run.status == "done" else "🛑 ") + md_safe(run.final)
        )
    if run.outbox or run.tickets:
        with st.expander(f"📤 Side effects: {len(run.outbox)} email(s), {len(run.tickets)} ticket(s)", expanded=True):
            for m in run.outbox:
                st.markdown(f"**Email to** `{m['to']}`: **{md_safe(html.escape(m['subject']))}**")
                st.caption(md_safe(m["body"]))
            for t in run.tickets:
                st.markdown(f"**Ticket** {t['id']} ({t['priority']}): {md_safe(html.escape(t['title']))}")
    st.caption(f"Model calls for this run: {run.calls}")

    st.markdown("""
**What to look for**
- **Multi-step:** does it search before stating facts, and use the calculator rather than doing maths in its head?
- **Blocked:** the allowlist stops the email *even if you'd approve it*. Hard policy sits above human approval.
- **Never ask:** switch the policy and rerun. The email goes out with no checkpoint. Would you ship that?
- **Overkill:** a fixed calculator call answers it. The agent spends extra model calls to get there.
""")


def _advance(rag, run: ag.AgentRun) -> None:
    with st.spinner("Agent thinking…"):
        ag.advance(run, rag, lambda system, prompt: call_model(rag, run.model, system, prompt))


# ── Lab B: compounding errors ─────────────────────────────────

def _lab_compounding() -> None:
    st.markdown("Per-step accuracy looks high until you chain steps together.")
    c1, c2 = st.columns(2)
    with c1:
        p = st.slider("Per-step success rate", 0.80, 0.999, 0.95, 0.005, key="ch5_p", format="%.3f")
    with c2:
        catch = st.slider("Share of errors a human reviewer catches at a checkpoint", 0, 100, 80, key="ch5_c") / 100
    xs = list(range(1, 21))
    plain = [p ** n for n in xs]
    # simplified: a review after each step catches a share of that step's errors
    reviewed = [(1 - (1 - p) * (1 - catch)) ** n for n in xs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=plain, name="No review", line=dict(color="#ff6b6b", width=3)))
    fig.add_trace(go.Scatter(x=xs, y=reviewed, name="Human checkpoints", line=dict(color="#3effa3", width=3)))
    fig.add_hline(y=0.5, line_dash="dot", annotation_text="coin flip")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Steps in the task",
                      yaxis_title="Chance the whole task succeeds", yaxis_range=[0, 1.02],
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
    half = next((n for n, v in zip(xs, plain) if v < 0.5), None)
    st.markdown(
        f"At **{p:.1%}** per step, a 10-step task succeeds **{p ** 10:.0%}** of the time"
        + (f", and it drops below a coin flip after **{half} steps**." if half else ".")
        + f" Checkpoints that catch {catch:.0%} of errors lift a 10-step task to **{reviewed[9]:.0%}**. "
        "That's why approval gates go on the costly, irreversible steps."
    )


# ── Lab C: when not to use an agent ───────────────────────────

_DRILL = [
    ("Convert every invoice PDF in a folder to a spreadsheet row, same fields each time.", "Workflow",
     "The steps never change. A fixed pipeline (extract → validate → write) is cheaper, faster and testable."),
    ("Research a new market: search the web, read sources, decide what to look up next, write a brief.", "Agent",
     "The next step depends on what was found. That open-ended path is what agents are for, with a human reviewing the brief."),
    ("Answer \"what's our refund policy?\" from the handbook.", "Single call (RAG)",
     "One retrieval and one answer. No loop, no tools beyond search."),
    ("Automatically refund customers who complain, up to $500, with no human review.", "Agent + human approval",
     "Money moves and can't easily be undone. Use an agent to prepare the refund and a human (or a strict cap and policy) to approve it."),
]


def _lab_drill() -> None:
    st.markdown("Pick the **least autonomy** that solves each problem.")
    opts = ["Single call (RAG)", "Workflow", "Agent", "Agent + human approval"]
    for i, (s, a, why) in enumerate(_DRILL, 1):
        with st.container(border=True):
            st.markdown(f"**{i}.** {s.replace('$', chr(92) + '$')}")
            pick = st.radio("Your call", opts, key=f"ch5_drill_{i}", horizontal=True, index=None)
            if pick:
                (st.success if pick == a else st.warning)(
                    ("✅ " if pick == a else f"Better: **{a}**. ") + why.replace("$", "\\$")
                )


def lab(rag) -> None:
    ta, tb, tc = st.tabs(["A · Run an agent", "B · Errors compound", "C · When not to"])
    with ta:
        if not needs_index(rag):
            _lab_agent(rag)
    with tb:
        _lab_compounding()
    with tc:
        _lab_drill()


def interview() -> None:
    interview_card(
        question="How would you design safeguards for an AI system that can take actions on a user's behalf?",
        asked_by="Common at Google, Microsoft, OpenAI-style product roles",
        structure=[
            "Scope the actions and classify them: read-only, reversible, irreversible, plus the cost of a mistake.",
            "Explain why risk compounds over multi-step tasks, using the per-step accuracy math.",
            "Layer the safeguards: least privilege, hard policy limits, human approval on irreversible steps, budgets.",
            "Defend against prompt injection: tool and web content is data, never instructions.",
            "Make it observable and reversible: audit log, undo windows, easy escalation.",
            "Earn autonomy: start with approval on everything, widen it as evals show low error rates.",
        ],
        sample=(
            "I'd start by listing what the agent can do and sorting actions by reversibility and cost of "
            "error. Reading email is low risk; sending it, paying or deleting is high. Because errors "
            "compound (95% per step is about 60% over ten steps), I wouldn't let the agent take "
            "irreversible actions unattended at launch. It drafts, and the user approves with one tap. "
            "Under that sit hard limits no model output can override: spending caps, allowlisted "
            "recipients, no deletion of shared data. The agent gets least-privilege credentials, treats "
            "anything it reads as untrusted to defend against prompt injection, and logs every step. "
            "I'd measure task success and wrong-action rate, and relax approvals per action type only "
            "once that rate is proven low, for example auto-sending replies to known contacts."
        ),
        change_my_mind=(
            "For purely read-only agents (research, summarising), I'd drop most approval gates and "
            "focus on citations and accuracy evals instead. The irreversible-action risk isn't there."
        ),
        pitfalls=[
            "Treating every action the same instead of tiering by reversibility.",
            "Relying only on the model's judgement, with no hard limits.",
            "Forgetting prompt injection from tool outputs.",
            "Never planning how autonomy expands as trust is earned.",
        ],
    )


CHAPTER = {
    "id": "ch5",
    "title": "Agents and safeguards",
    "short": "Agents",
    "tagline": "LLMs that take actions: why errors compound, and how to keep humans in control.",
    "concept": concept,
    "lab": lab,
    "interview": interview,
}
