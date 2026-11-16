# ============================================================
# Chapter 4 — Latency and cost
#
# Labs (one per lever):
#   A. Streaming     — perceived wait (first word) vs total time
#   B. Caching       — exact + semantic cache in front of the model
#   C. Routing       — easy questions → small model, hard → large
#   D. Build vs buy  — API vs self-hosted cost, break-even volume
# ============================================================

from __future__ import annotations

import html
import math
import re
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from config import MODELS
from learn.common import answer_box, call_model, html_safe, interview_card, md_safe, model_picker, needs_index

# Illustrative list prices (USD per 1M tokens). Real prices vary by provider;
# the free models in this app cost $0 but are rate-limited.
PRICE_SMALL = {"in": 0.05, "out": 0.10}
PRICE_LARGE = {"in": 1.25, "out": 5.00}

HARD_WORDS = re.compile(
    r"\b(why|how|compare|comparison|explain|analy[sz]e|trade.?offs?|pros|cons|recommend\w*|"
    r"strategy|impact|implications?|summar\w*|evaluate|steps|plan|risks?)\b",
    re.I,
)


def concept() -> None:
    st.markdown("""
Latency and cost usually come up as a **follow-up** ("would that eat more tokens?"). You don't need GPU internals. You do need the **levers** and the **math**.

**Where the time goes in one answer**
1. **Network + queue:** reaching the provider and waiting for a free GPU.
2. **Prefill:** the model reads your prompt. It grows with prompt length, which is why RAG's few chunks are faster than a whole document.
3. **Thinking** (reasoning models only): hidden tokens before the answer.
4. **Generation:** the answer, token by token. It grows with answer length.

**Time to first token (TTFT)** is what users *feel*. **Total time** is what your servers pay for.

**Where the money goes:** `cost = calls × (input tokens × input price + output tokens × output price)`. Output tokens usually cost 3–5× more than input tokens.

**The levers**

| Lever | What it does | Saves | Trade-off |
|---|---|---|---|
| **Streaming** | Show words as they're generated | Perceived latency | None on cost. Total time is unchanged |
| **Caching** | Reuse answers to repeated (or similar) questions | Cost and latency on cache hits | Stale or wrong answers if the match is loose |
| **Routing** | Send easy questions to a small model | Most of the cost | Quality drops if the router misjudges |
| **Shorter prompts** | Fewer or smaller chunks, tighter instructions | Cost and prefill time | Risk of dropping needed context |
| **Batching** | Group non-urgent requests; batch APIs are often ~50% off | Cost and throughput | Results take minutes to hours |
| **Smaller or distilled model** | Fine-tune a small model for a narrow task | Cost and latency at scale | Upfront work, needs evals |

**Build vs buy:** APIs cost per token, so cost rises with usage. Self-hosting costs per GPU-hour plus engineers, roughly fixed until you need more GPUs. APIs win at low or spiky volume; self-hosting can win at high, steady volume, or when data can't leave your network. Lab D shows the break-even point.
""")


# ── Lab A: streaming ──────────────────────────────────────────

def _lab_streaming(rag) -> None:
    st.markdown("Watch the answer **stream in**, and compare *when the first word appeared* with *when it finished*.")
    model = model_picker("Model", key="ch4s_model", prefer="nvidia/nemotron-3.5-lightning:free")
    q = st.text_input("Question", "Summarize the key points of the document in 5 bullet points.", key="ch4s_q")
    st.caption("**1 model call.** Streaming works for cloud models. Local models show the answer at the end.")
    if st.button("▶️ Stream it", type="primary", key="ch4s_run", disabled=not (model and q.strip())):
        chunks = rag.retrieve(q, top_k=5)
        system, prompt = rag.build_prompt(q, chunks)
        timer = st.empty()
        box = st.empty()
        t0 = time.perf_counter()
        first_word: list[float] = []

        def on_token(text: str) -> None:
            if not first_word:
                first_word.append(time.perf_counter() - t0)
            timer.caption(f"⏱️ first word after {first_word[0]:.1f}s · streaming… {time.perf_counter() - t0:.1f}s")
            box.markdown(f'<div class="answer-box">{html_safe(text)}</div>', unsafe_allow_html=True)

        box.markdown('<div class="answer-box">⏳ waiting for the first token…</div>', unsafe_allow_html=True)
        res = call_model(rag, model, system, prompt, on_token=on_token)
        timer.empty()
        box.empty()
        st.session_state.ch4s = res

    res = st.session_state.get("ch4s")
    if res is None:
        return
    answer_box("Answer", res)
    if res.error:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("First token (incl. thinking)", f"{res.ttft_ms / 1000:.1f}s")
    c2.metric("First visible word", f"{res.time_to_answer_ms / 1000:.1f}s")
    c3.metric("Complete answer", f"{res.wall_clock_s:.1f}s")
    saved = res.wall_clock_s - res.time_to_answer_ms / 1000
    st.info(
        f"With streaming the user starts reading after **{res.time_to_answer_ms / 1000:.1f}s**. Without it "
        f"they'd stare at a spinner for **{res.wall_clock_s:.1f}s**, {saved:.1f}s longer. Streaming "
        "changes nothing about cost or total compute; it's purely a UX lever."
        + (f" Note the gap between first token and first word: {res.reasoning_tokens} hidden thinking "
           "tokens. Reasoning models feel slower even when streaming."
           if res.reasoning_tokens else ""),
        icon="💡",
    )


# ── Lab B: caching ────────────────────────────────────────────

def _cache() -> list[dict]:
    return st.session_state.setdefault("ch4_cache", [])


def _embed(rag, text: str) -> np.ndarray:
    v = np.asarray(rag.embed_client.embed(text), dtype=float)
    return v / (np.linalg.norm(v) or 1.0)


def _lab_caching(rag) -> None:
    st.markdown(
        "A cache sits **in front of** the model. Ask a question, then ask it again, then ask a "
        "**paraphrase**. An *exact* cache only catches identical text; a *semantic* cache matches "
        "similar meaning using embeddings."
    )
    model = model_picker("Model (used on a cache miss)", key="ch4c_model", prefer="recursive/flint-v1.0:free")
    c1, c2 = st.columns([3, 1])
    with c1:
        q = st.text_input("Question", "What are the main risks mentioned?", key="ch4c_q")
    with c2:
        sim_thr = st.slider("Semantic match ≥", 0.70, 0.99, 0.90, 0.01, key="ch4c_thr",
                            help="Higher = safer but fewer hits. Too low and different questions get the same answer.")
    st.caption("**0 calls on a hit, 1 on a miss.** Try: ask → ask again → reword it → ask something different.")

    if st.button("▶️ Ask (through the cache)", type="primary", key="ch4c_run", disabled=not (model and q.strip())):
        t0 = time.perf_counter()
        cache = _cache()
        norm = " ".join(q.lower().split())
        hit, kind, score = None, "", 0.0
        for e in cache:
            if e["norm"] == norm:
                hit, kind, score = e, "exact", 1.0
                break
        qv = _embed(rag, q)
        if hit is None and cache:
            sims = [float(qv @ e["vec"]) for e in cache]
            best = int(np.argmax(sims))
            if sims[best] >= sim_thr:
                hit, kind, score = cache[best], "semantic", sims[best]
        if hit:
            st.session_state.ch4c_last = dict(q=q, hit=True, kind=kind, score=score, entry=hit,
                                              ms=(time.perf_counter() - t0) * 1000)
            hit["hits"] += 1
        else:
            chunks = rag.retrieve(q, top_k=5)
            system, prompt = rag.build_prompt(q, chunks)
            res = call_model(rag, model, system, prompt)
            entry = dict(q=q, norm=norm, vec=qv, answer=res.response or f"❌ {res.error}",
                         tokens=res.prompt_tokens + res.output_tokens, hits=0)
            if not res.error:
                cache.append(entry)
            st.session_state.ch4c_last = dict(q=q, hit=False, kind="miss", score=0.0, entry=entry,
                                              ms=res.wall_clock_s * 1000)

    last = st.session_state.get("ch4c_last")
    if last:
        if last["hit"]:
            st.success(
                f"**Cache hit ({last['kind']}, similarity {last['score']:.2f})** in {last['ms']:.0f} ms. "
                f"No model call, ~{last['entry']['tokens']:,} tokens saved. Matched: “{md_safe(last['entry']['q'])}”"
            )
        else:
            st.warning(f"**Cache miss**: called the model ({last['ms'] / 1000:.1f}s). Answer stored for next time.")
        st.markdown(f'<div class="answer-box">{html_safe(last["entry"]["answer"])}</div>', unsafe_allow_html=True)

    cache = _cache()
    if cache:
        hits = sum(e["hits"] for e in cache)
        saved = sum(e["hits"] * e["tokens"] for e in cache)
        st.caption(f"Cache: {len(cache)} entries · {hits} hits · ~{saved:,} tokens saved")
        if st.button("Clear cache", key="ch4c_clear"):
            st.session_state.ch4_cache = []
            st.session_state.pop("ch4c_last", None)
            st.rerun()
    st.markdown("""
**What to look for**
- Lower the threshold until a *different* question gets a cached answer. That's the danger of semantic caching.
- Caches go stale: when the document changes, cached answers don't. Real systems expire entries or clear the cache on re-index.
""")


# ── Lab C: routing ────────────────────────────────────────────

def _route(q: str) -> tuple[str, list[str]]:
    reasons = []
    words = len(q.split())
    if words > 15:
        reasons.append(f"long question ({words} words)")
    found = sorted({m.lower() for m in HARD_WORDS.findall(q)})
    if found:
        reasons.append("reasoning words: " + ", ".join(found))
    return ("large" if reasons else "small"), reasons or ["short factual lookup"]


def _cost(res, price) -> float:
    return (res.prompt_tokens * price["in"] + res.output_tokens * price["out"]) / 1e6


def _lab_routing(rag) -> None:
    st.markdown(
        "A **router** sends each question to the cheapest model that can handle it. Here a simple "
        "rule decides; real routers often use a small classifier model. Run the question on **both** "
        "models to see what routing gains or loses."
    )
    c1, c2 = st.columns(2)
    with c1:
        small = model_picker("Small / cheap model", key="ch4r_small", prefer="liquid/lfm-3-1.7b:free")
    with c2:
        large = model_picker("Large / capable model", key="ch4r_large", prefer="nvidia/nemotron-3.5-lightning:free")
    q = st.text_input("Question", "Who is the CEO?", key="ch4r_q")
    tier, reasons = _route(q)
    st.markdown(f"**Router decision:** → **{tier} model** ({'; '.join(reasons)})")
    st.caption(
        "**2 model calls** (both models, so you can compare). Try “Who is the CEO?” then "
        "“Compare the risks and recommend which priority matters most, and why.”"
    )

    if st.button("▶️ Run on both", type="primary", key="ch4r_run", disabled=not (small and large and q.strip())):
        chunks = rag.retrieve(q, top_k=5)
        system, prompt = rag.build_prompt(q, chunks)
        with st.spinner("Small model…"):
            rs = call_model(rag, small, system, prompt)
        with st.spinner("Large model…"):
            rl = call_model(rag, large, system, prompt)
        st.session_state.ch4r = (q, tier, rs, rl)

    if st.session_state.get("ch4r"):
        q0, tier0, rs, rl = st.session_state.ch4r
        a, b = st.columns(2)
        with a:
            answer_box(f"Small · {MODELS.get(small, {}).get('label', small)}"
                       + ("  ← router's pick" if tier0 == "small" else ""), rs)
            if not rs.error:
                st.caption(f"💲 ${_cost(rs, PRICE_SMALL):.6f} at illustrative small-model prices")
        with b:
            answer_box(f"Large · {MODELS.get(large, {}).get('label', large)}"
                       + ("  ← router's pick" if tier0 == "large" else ""), rl)
            if not rl.error:
                st.caption(f"💲 ${_cost(rl, PRICE_LARGE):.6f} at illustrative large-model prices")
        st.caption("Was the small model's answer good enough? That judgement, at scale, is what a router eval measures.")

    st.markdown("#### Blended cost at scale")
    d1, d2, d3 = st.columns(3)
    with d1:
        qpd = st.number_input("Questions per day", 100, 10_000_000, 100_000, step=10_000, key="ch4r_qpd")
    with d2:
        share_small = st.slider("Share routed to small model", 0, 100, 70, key="ch4r_share") / 100
    with d3:
        tok_in, tok_out = 800, 250
        st.caption(f"Assumes {tok_in} input + {tok_out} output tokens per question, illustrative prices: "
                   f"small \\${PRICE_SMALL['in']}/\\${PRICE_SMALL['out']}, large \\${PRICE_LARGE['in']}/\\${PRICE_LARGE['out']} per 1M.")
    per_q = lambda p: (tok_in * p["in"] + tok_out * p["out"]) / 1e6
    all_large = qpd * 30 * per_q(PRICE_LARGE)
    blended = qpd * 30 * (share_small * per_q(PRICE_SMALL) + (1 - share_small) * per_q(PRICE_LARGE))
    m1, m2, m3 = st.columns(3)
    m1.metric("All large model / month", f"${all_large:,.0f}")
    m2.metric("Routed / month", f"${blended:,.0f}")
    m3.metric("Saving", f"{(1 - blended / all_large):.0%}" if all_large else "—")


# ── Lab D: build vs buy ───────────────────────────────────────

def _lab_build_buy() -> None:
    st.markdown(
        "Compare paying an **API per token** with **self-hosting** an open model on rented GPUs. "
        "All inputs are editable; defaults are rough, illustrative numbers."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Workload**")
        qpd = st.number_input("Questions per day", 100, 50_000_000, 200_000, step=50_000, key="ch4b_qpd")
        tin = st.number_input("Input tokens / question", 50, 50_000, 1_000, step=100, key="ch4b_in")
        tout = st.number_input("Output tokens / question", 10, 10_000, 300, step=50, key="ch4b_out")
    with c2:
        st.markdown("**API**")
        pin = st.number_input("Input $ / 1M tokens", 0.0, 100.0, 0.50, step=0.05, key="ch4b_pin")
        pout = st.number_input("Output $ / 1M tokens", 0.0, 200.0, 1.50, step=0.10, key="ch4b_pout")
    with c3:
        st.markdown("**Self-host**")
        gpu_hr = st.number_input("GPU $ / hour", 0.1, 50.0, 2.50, step=0.25, key="ch4b_gpu")
        tps = st.number_input("Output tokens/sec per GPU (batched)", 50, 20_000, 1_500, step=100, key="ch4b_tps",
                              help="Throughput with many requests batched together, not single-user speed.")
        util = st.slider("Average utilisation", 10, 90, 40, key="ch4b_util",
                         help="Traffic is spiky; GPUs sit partly idle.") / 100
        min_gpus = st.number_input("Minimum GPUs (redundancy)", 1, 64, 2, key="ch4b_min")
        eng = st.number_input("Engineering + ops $ / month", 0, 1_000_000, 15_000, step=1_000, key="ch4b_eng")

    def api_cost(q):
        return q * 30 * (tin * pin + tout * pout) / 1e6

    def self_cost(q):
        tokens_per_sec = q * tout / 86_400
        gpus = max(min_gpus, math.ceil(tokens_per_sec / (tps * util)))
        return gpus * gpu_hr * 730 + eng, gpus

    api = api_cost(qpd)
    selfc, gpus = self_cost(qpd)
    m1, m2, m3 = st.columns(3)
    m1.metric("API / month", f"${api:,.0f}")
    m2.metric("Self-host / month", f"${selfc:,.0f}", help=f"{gpus} GPU(s) + engineering")
    m3.metric("Cheaper option", "API" if api <= selfc else "Self-host")

    xs = np.unique(np.geomspace(1_000, 20_000_000, 120).astype(int))
    api_y = [api_cost(x) for x in xs]
    self_y = [self_cost(x)[0] for x in xs]
    be = next((x for x, a, s in zip(xs, api_y, self_y) if a > s), None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=api_y, name="API", line=dict(color="#6C8EF5", width=3)))
    fig.add_trace(go.Scatter(x=xs, y=self_y, name="Self-host", line=dict(color="#ffb347", width=3, shape="hv")))
    fig.add_vline(x=qpd, line_dash="dash", annotation_text="your volume")
    if be:
        fig.add_vline(x=be, line_dash="dot", line_color="#3effa3", annotation_text="break-even",
                      annotation_position="bottom right")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_type="log", yaxis_type="log",
                      xaxis_title="Questions per day (log)", yaxis_title="$ / month (log)",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        (f"Break-even at roughly **{be:,} questions/day**. " if be else
         "With these inputs the API stays cheaper across the whole range. ")
        + "Self-hosting is a step function (whole GPUs plus a fixed team), while the API grows "
        "smoothly with use. Remember the non-cost reasons too: data residency, latency control, "
        "and model choice."
    )


def lab(rag) -> None:
    ta, tb, tc, td = st.tabs(["A · Streaming", "B · Caching", "C · Routing", "D · Build vs buy"])
    with ta:
        if not needs_index(rag):
            _lab_streaming(rag)
    with tb:
        if not needs_index(rag):
            _lab_caching(rag)
    with tc:
        if not needs_index(rag):
            _lab_routing(rag)
    with td:
        _lab_build_buy()


def interview() -> None:
    interview_card(
        question="Would your prototype eat more tokens? How would you bring the cost down?",
        asked_by="Meta (follow-up)",
        structure=[
            "Estimate first: calls per task × tokens per call × price. Say the number out loud.",
            "Name where the tokens go: long prompts, retrieved context, output length, retries, agent loops.",
            "Pick levers in order of effort: shorter prompts and caching, then routing, then smaller or tuned models.",
            "Protect quality: every lever is checked against the eval set.",
            "Mention latency separately: streaming fixes perceived wait at no cost.",
        ],
        sample=(
            "Yes. A RAG answer sends about 1,000 tokens of context plus the question, so input tokens "
            "dominate. At 100k questions a day on a mid-tier model that's a few thousand dollars a "
            "month. I'd cut it in three steps. First, trim the prompt: fewer, better chunks and a "
            "tighter system prompt, which often saves 30–50%. Second, cache repeated questions, which "
            "is common in support. Third, route the easy majority to a small model and keep the large "
            "model for hard queries. Each step has to hold our answer-quality score on the eval set, or "
            "it doesn't ship. For latency, I'd stream responses so time-to-first-word stays under a second."
        ),
        change_my_mind=(
            "If volume were tiny (an internal tool with 200 users), I wouldn't optimise cost at all. "
            "I'd use the best model and spend the effort on quality."
        ),
        pitfalls=[
            "Talking about levers without estimating the cost first.",
            "Forgetting agent loops and retries, which multiply calls.",
            "Optimising cost with no quality guardrail.",
            "Confusing perceived latency (streaming) with total compute.",
        ],
    )


CHAPTER = {
    "id": "ch4",
    "title": "Latency and cost",
    "short": "Latency & cost",
    "tagline": "Where time and money go in an LLM answer, and the levers that cut them.",
    "concept": concept,
    "lab": lab,
    "interview": interview,
}
