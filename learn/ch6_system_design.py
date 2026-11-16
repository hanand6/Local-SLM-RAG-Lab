# ============================================================
# Chapter 6 — AI system design (boxes and arrows)
#
# Lab A: worked example — this app's own architecture, each box
#        tied to the requirement it serves.
# Lab B: design canvas — scope → pick components (each tied to a
#        requirement) → auto-drawn diagram → optional AI critique.
# ============================================================

from __future__ import annotations

import streamlit as st

from learn.common import call_model, interview_card, md_safe, model_picker

# Canonical request path (order matters for drawing arrows) + side systems.
COMPONENTS = {
    # key: (label, lane, default requirement it serves)
    "client":      ("Client app (web / mobile)", "path", "Users ask questions and see answers"),
    "gateway":     ("API gateway: auth + rate limits", "path", "Security, abuse protection, quotas"),
    "in_guard":    ("Input guardrails (PII, injection, policy)", "path", "Safety and privacy before the model sees input"),
    "cache":       ("Response cache", "path", "Latency and cost for repeated questions"),
    "router":      ("Query understanding + model router", "path", "Send easy queries to cheap models; detect intent"),
    "retrieval":   ("Retrieval (search / RAG)", "path", "Fresh, citable facts; fewer hallucinations"),
    "llm":         ("LLM (generation)", "path", "Produce the answer"),
    "out_guard":   ("Output guardrails (safety, grounding check)", "path", "Block unsafe or unsupported answers"),
    "tools":       ("Tools / actions (APIs)", "side", "Let the system do things, not just answer"),
    "index":       ("Index + vector store", "side", "Searchable knowledge for retrieval"),
    "ingest":      ("Ingestion pipeline (crawl, chunk, embed)", "side", "Keep the index fresh"),
    "human":       ("Human handoff / approval", "side", "Fallback when confidence is low or actions are risky"),
    "logging":     ("Logging + monitoring", "side", "Debugging, latency/cost dashboards, audit trail"),
    "evals":       ("Offline evals (golden set)", "side", "Catch regressions before shipping"),
    "feedback":    ("User feedback loop (thumbs, reports)", "side", "Online quality signal, new eval cases"),
}
PATH_ORDER = [k for k, v in COMPONENTS.items() if v[1] == "path"]

SIDE_EDGES = [  # (from, to, label) — drawn only when both ends are chosen
    ("retrieval", "index", "query"), ("ingest", "index", "update"), ("llm", "tools", "call"),
    ("router", "human", "low confidence"), ("out_guard", "human", "flagged"),
    ("llm", "logging", "traces"), ("client", "feedback", "👍/👎"),
    ("feedback", "evals", "new cases"), ("logging", "evals", "samples"),
]

PROMPTS = {
    "Gemini answering a user query": {
        "brief": "Design a high-level system for an assistant like Gemini responding to a user's query.",
        "hints": "Consumer scale (100M+ users), mixed queries (chit-chat, facts, fresh news, tasks), "
                 "latency target under 1–2s to first word, safety at scale, cost per query matters.",
        "suggest": ["client", "gateway", "in_guard", "cache", "router", "retrieval", "llm", "out_guard",
                    "index", "ingest", "logging", "evals", "feedback"],
    },
    "Support copilot for agents": {
        "brief": "Design a copilot that drafts replies for customer-support agents from the help center and past tickets.",
        "hints": "~2,000 support agents, the human always sends, answers must cite sources, "
                 "PII in tickets, measured on handle time and CSAT.",
        "suggest": ["client", "gateway", "in_guard", "retrieval", "llm", "out_guard", "index", "ingest",
                    "human", "logging", "evals", "feedback"],
    },
    "Contract Q&A for a law firm": {
        "brief": "Design document Q&A over confidential contracts for a law firm.",
        "hints": "200 lawyers, highly confidential data (may require self-hosting), accuracy over speed, "
                 "every claim must be cited, audit trail required.",
        "suggest": ["client", "gateway", "in_guard", "retrieval", "llm", "out_guard", "index", "ingest",
                    "human", "logging", "evals"],
    },
}


def concept() -> None:
    st.markdown("""
AI system design questions ("design Gemini answering a query") test **structured thinking**, not code. Stay at the level of **boxes and arrows**, and tie every box to a requirement.

**A reliable 5-step structure (about 30 minutes)**
1. **Clarify and scope (5 min).** Who are the users? What's the core use case? What scale (users, queries per second)? What are the constraints: latency, cost, privacy, regulation? How will success be measured?
2. **Requirements.** Turn the answers into a short list: functional (*answer with citations*) and non-functional (*P95 < 2s*, *no PII leaves the region*).
3. **High-level design.** Draw the request path from left to right, plus the side systems. **Every box must point back to a requirement.** If it doesn't, cut it.
4. **Deep-dive 1–2 areas** where the interesting trade-offs are: retrieval quality, the model-routing policy, the safety layer.
5. **Evaluate and iterate.** Name the offline metrics, the online metrics, the rollout plan and the failure modes.

**The usual building blocks of an LLM product**

| Request path | Side systems |
|---|---|
| Client → gateway (auth, rate limits) | Ingestion pipeline → index / vector store |
| Input guardrails (PII, prompt injection) | Tools / actions |
| Cache → router (pick a model by difficulty) | Human handoff / approval |
| Retrieval → LLM → output guardrails | Logging, monitoring, evals, feedback loop |

**What interviewers listen for**
- You **scoped first** rather than drawing immediately.
- Each component is **justified by a requirement**. Not a buzzword list.
- You name **trade-offs** (cache freshness vs cost, a bigger model vs latency) and choose.
- You close with **how you'd measure it** and **how you'd launch it**.
""")


def _dot(chosen: list[str], notes: dict[str, str], title: str) -> str:
    lines = [
        # Top-to-bottom: the request path runs down, side systems sit beside it.
        # (Left-to-right gets squashed unreadably thin with 10+ boxes.)
        "digraph G {", 'rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.35;',
        'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 '
        'fillcolor="#16213e" fontcolor="#e0e0e0" color="#0f3460"];',
        'edge [color="#8892b0" fontcolor="#8892b0" fontsize=9 fontname="Helvetica"];',
        f'labelloc="t"; label="{title}"; fontcolor="#a0c4ff"; fontname="Helvetica";',
    ]
    for k in chosen:
        label, lane, _ = COMPONENTS[k]
        req = notes.get(k, "").replace('"', "'")
        text = f"{label}\\n— {req[:48]}" if req else label
        color = "#1a4731" if lane == "path" else "#2a2342"
        lines.append(f'{k} [label="{text}" fillcolor="{color}"];')
    path = [k for k in PATH_ORDER if k in chosen]
    for a, b in zip(path, path[1:]):
        lines.append(f"{a} -> {b};")
    if "client" in chosen and path and path[-1] != "client":
        lines.append(f'{path[-1]} -> client [label="answer" style=dashed constraint=false];')
    for a, b, lbl in SIDE_EDGES:
        if a in chosen and b in chosen:
            lines.append(f'{a} -> {b} [label="{lbl}" style=dashed];')
    lines.append("}")
    return "\n".join(lines)


# ── Lab A: worked example ─────────────────────────────────────

def _lab_example() -> None:
    st.markdown("**Worked example: this app.** Every box maps to a requirement you've seen in earlier chapters.")
    chosen = ["client", "router", "retrieval", "llm", "index", "ingest", "logging", "evals"]
    notes = {
        "client": "Streamlit UI: ask, compare, learn",
        "router": "Pick model per tab (free OpenRouter / Ollama)",
        "retrieval": "Top-k chunks + optional BM25 rerank",
        "llm": "Free models; retries on 429",
        "index": "ChromaDB in memory (local)",
        "ingest": "Load → chunk → embed (MiniLM, local)",
        "logging": "Latency, tokens, retries per call",
        "evals": "RAGAS tab + Chapter 3 golden set",
    }
    st.graphviz_chart(_dot(chosen, notes, "SLM RAG Lab"), use_container_width=True)
    st.markdown("""
| Requirement | How this app meets it | What's missing for production |
|---|---|---|
| Answers grounded in documents | Retrieval + strict prompt + citations | Output grounding check before showing answers |
| $0 inference cost | Free models, local embeddings | Rate limits: needs a paid tier or self-hosting to scale |
| Privacy | Local embeddings and vector store | Cloud models see retrieved chunks: a privacy review, or local models |
| Know when quality drops | RAGAS and the golden-set lab | Automated evals on every change; online feedback |
| Many users | ❌ single user, in-memory index | Gateway, auth, persistent vector DB, cache, monitoring |
""".replace("$0", "\\$0"))
    st.caption("Spotting what's missing is the point: in an interview, the 'what I'd add for production' column is gold.")


# ── Lab B: design canvas ──────────────────────────────────────

def _lab_canvas(rag) -> None:
    pick = st.radio("Design prompt", list(PROMPTS), horizontal=True, key="ch6_prompt")
    p = PROMPTS[pick]
    st.markdown(f"#### 📝 {p['brief']}")
    with st.expander("Interviewer's answers to your clarifying questions (open after you've written your own)"):
        st.markdown(p["hints"])

    st.markdown("**Step 1: clarify and scope**")
    c1, c2 = st.columns(2)
    with c1:
        users = st.text_area("Users and core use case", key=f"ch6_users_{pick}", height=80)
        scale = st.text_area("Scale (users, queries/sec, data size)", key=f"ch6_scale_{pick}", height=80)
    with c2:
        constraints = st.text_area("Constraints (latency, cost, privacy, regulation)", key=f"ch6_cons_{pick}", height=80)
        success = st.text_area("Success metrics", key=f"ch6_succ_{pick}", height=80)

    st.markdown("**Step 2: pick components, and tie each to a requirement**")
    comp_key = f"ch6_comp_{pick}"
    chosen = st.multiselect(
        "Components", list(COMPONENTS),
        format_func=lambda k: COMPONENTS[k][0], key=comp_key,
        help="Start small. Add a component only when you can say which requirement it serves.",
    )
    # A callback runs before the next render, so it may set the widget's value.
    st.button("Fill in a reference set of components", key=f"ch6_fill_{pick}",
              on_click=lambda: st.session_state.update({comp_key: list(p["suggest"])}))

    notes: dict[str, str] = {}
    if chosen:
        cols = st.columns(2)
        for i, k in enumerate(chosen):
            with cols[i % 2]:
                notes[k] = st.text_input(
                    f"{COMPONENTS[k][0]}: serves which requirement?",
                    value=COMPONENTS[k][2], key=f"ch6_note_{pick}_{k}",
                )
        st.markdown("**Step 3: the diagram** (green = request path, purple = side systems)")
        st.graphviz_chart(_dot(chosen, notes, pick), use_container_width=True)

        must = {"logging", "evals"} - set(chosen)
        if must:
            st.warning("No " + " or ".join(COMPONENTS[m][0].split(" (")[0] for m in sorted(must))
                       + " in your design. How will you know it works after launch?")

    st.markdown("**Step 4: get a critique** (optional, **1 model call**)")
    model = model_picker("Reviewer model", key="ch6_model", prefer="nvidia/nemotron-3.5-lightning:free")
    if st.button("🧐 Critique my design", key=f"ch6_crit_{pick}", disabled=not (model and chosen)):
        design = "\n".join(f"- {COMPONENTS[k][0]}: serves '{notes.get(k, '')}'" for k in chosen)
        prompt = (
            f"Design prompt: {p['brief']}\n\n"
            f"Candidate's scoping:\n- Users/use case: {users or '(blank)'}\n- Scale: {scale or '(blank)'}\n"
            f"- Constraints: {constraints or '(blank)'}\n- Success metrics: {success or '(blank)'}\n\n"
            f"Components (with the requirement each serves):\n{design}\n\n"
            "Review this like a senior interviewer in a product-management AI system design interview. "
            "Be specific and brief. Give: (1) a score out of 5 for scoping, requirement-linkage, "
            "completeness and trade-offs; (2) the two most important missing pieces or weak links; "
            "(3) one trade-off the candidate should discuss; (4) one follow-up question you'd ask."
        )
        with st.spinner("Reviewing…"):
            res = call_model(rag, model, "You are an experienced, fair, direct interviewer.", prompt)
        st.session_state[f"ch6_review_{pick}"] = res.response or f"❌ {res.error}"
    if st.session_state.get(f"ch6_review_{pick}"):
        with st.container(border=True):
            st.markdown(md_safe(st.session_state[f"ch6_review_{pick}"]))


def lab(rag) -> None:
    ta, tb = st.tabs(["A · Worked example", "B · Design canvas"])
    with ta:
        _lab_example()
    with tb:
        _lab_canvas(rag)


def interview() -> None:
    interview_card(
        question="Design a high-level system for Gemini responding to a user query.",
        asked_by="Google",
        structure=[
            "Clarify: users, query types, scale, latency target, safety bar, success metric. Then state your assumptions.",
            "Requirements: functional (answer, cite, act) and non-functional (P95 latency, cost per query, safety).",
            "Request path: client → gateway → input safety → cache → router → retrieval/tools → LLM → output safety → client.",
            "Side systems: index and ingestion for fresh facts, logging, evals, feedback loop.",
            "Deep-dive one area (e.g. the router or grounding) with a trade-off and your choice.",
            "Close with metrics and launch: offline evals, online metrics, staged rollout.",
        ],
        sample=(
            "I'd first scope it: consumer users on mobile, mixed queries from chit-chat to fresh facts to "
            "tasks, very high QPS, a target of under a second to first word, and success measured by "
            "satisfied sessions and a low rate of harmful or wrong answers. The request goes through a "
            "gateway for auth and quotas, then input safety checks for abuse and prompt injection. A cache "
            "serves popular queries. A router classifies intent: simple queries go to a small fast model; "
            "factual or fresh queries trigger search-based retrieval, so answers are grounded and cited; "
            "tasks call tools with user confirmation. The LLM's draft passes output safety and a grounding "
            "check before streaming back. On the side, ingestion keeps the search index fresh, and logging "
            "feeds dashboards and an eval set built from real failures. I'd deep-dive the router, because "
            "it drives both cost and quality, and launch it behind an experiment measuring satisfaction "
            "and cost per query."
        ),
        change_my_mind=(
            "If the scope were enterprise (company documents, strict permissions), I'd move access control "
            "into retrieval (document-level permissions) and make data residency a core requirement, which "
            "might mean dedicated or self-hosted models."
        ),
        pitfalls=[
            "Drawing before scoping.",
            "Listing components without saying which requirement each serves.",
            "Going too deep (pseudocode, GPU details) instead of boxes and arrows.",
            "No evaluation or launch plan at the end.",
        ],
    )


CHAPTER = {
    "id": "ch6",
    "title": "AI system design",
    "short": "System design",
    "tagline": "Boxes and arrows: scope first, tie every component to a requirement, then show how you'd measure it.",
    "concept": concept,
    "lab": lab,
    "interview": interview,
}
