# ============================================================
# Chapter 1 — Prompting vs RAG vs fine-tuning
#
# Lab: the same question three ways on your own document
#   A. Prompt only       (model's own knowledge)
#   B. Whole document    (long-context "stuff it all in")
#   C. RAG               (top-k retrieved chunks)
# Fine-tuning can't run on free tiers, so it gets a scenario quiz.
# ============================================================

from __future__ import annotations

import streamlit as st

from learn.common import answer_box, call_model, interview_card, model_picker, needs_index

LONG_CONTEXT_MAX_CHARS = 60_000   # ~15k tokens: fits every default model's context

SYSTEM_PLAIN = "You are a helpful assistant. Answer concisely."
SYSTEM_DOC = (
    "You are a precise document Q&A assistant. Answer only from the document provided. "
    "If the answer is not in it, say so. Be concise."
)


def concept() -> None:
    st.markdown("""
There are three ways to give a model the knowledge it needs. They differ in **where the knowledge lives**.

| | Prompting | RAG | Fine-tuning |
|---|---|---|---|
| Knowledge lives in | The prompt you write | An index, searched per question | The model's weights |
| Good for | Behaviour, format, small fixed facts | Large or changing facts, with citations | A consistent style, format or skill at scale |
| Freshness | Edit the prompt | Re-index the document (minutes) | Retrain (days), new eval cycle |
| Cost driver | Prompt length × calls | Retrieval + a few chunks per call | Training run + hosting a custom model |
| Can show its sources | Only if you pasted them | ✅ Yes, chunk-level | ❌ No |
| Main failure | Prompt too long, or facts missing | Wrong chunks retrieved | Learns style, not reliable facts; goes stale |

**Rules of thumb**
- Start with **prompting**. It's the cheapest to change, so get as far as you can with it.
- Add **RAG** when the facts are too big for the prompt, change often, or must be cited.
- Consider **fine-tuning** last: for behaviour you can't prompt reliably (a strict output format, a domain's tone) at high volume, where a smaller tuned model cuts cost and latency.
- **Long context** ("paste the whole document") is a fourth option now that context windows are 128k–1M tokens. It's simple and has no retrieval misses. But you pay for every token on every call, it's slower, and models get worse at finding facts in the middle of very long inputs.
- These aren't exclusive. Many products use **all three**: a tuned small model for format, RAG for facts, and a prompt for rules.
""")


def lab(rag) -> None:
    if needs_index(rag):
        return

    st.markdown(
        "Ask one question three ways and compare **answer quality, prompt size and speed**. "
        "Try a question only your document can answer, then a general-knowledge one."
    )
    model = model_picker("Model", key="ch1_model", prefer="recursive/flint-v1.0:free")
    question = st.text_input(
        "Question about your document",
        key="ch1_q",
        placeholder="e.g. What does the document recommend, and why?",
    )
    top_k = st.slider("RAG: chunks retrieved (top-k)", 2, 10, 5, key="ch1_k")

    doc_text, total_chars = rag.get_all_text(LONG_CONTEXT_MAX_CHARS)
    if total_chars > LONG_CONTEXT_MAX_CHARS:
        st.caption(
            f"ℹ️ Your indexed text is {total_chars:,} chars. The whole-document run sends the "
            f"first {LONG_CONTEXT_MAX_CHARS:,}, which shows the real limit of this approach."
        )
    st.caption("This lab makes **3 model calls**.")

    if st.button("▶️ Run all three", type="primary", key="ch1_run", disabled=not (model and question.strip())):
        with st.spinner("A. Prompt only..."):
            a = call_model(rag, model, SYSTEM_PLAIN, question)
        with st.spinner("B. Whole document..."):
            b = call_model(rag, model, SYSTEM_DOC, f"Document:\n{doc_text}\n\nQuestion: {question}\nAnswer:")
        with st.spinner("C. RAG..."):
            chunks = rag.retrieve(question, top_k=top_k)
            system, prompt = rag.build_prompt(question, chunks)
            c = call_model(rag, model, system, prompt)
        st.session_state.ch1_results = (question, a, b, c, chunks)

    if st.session_state.get("ch1_results"):
        q, a, b, c, chunks = st.session_state.ch1_results
        st.markdown(f"**Question:** {q}")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            answer_box("A. Prompt only", a, "No document. The model's own knowledge.")
        with col_b:
            answer_box("B. Whole document", b, "Everything pasted into the prompt.")
        with col_c:
            answer_box("C. RAG", c, f"Top {len(chunks)} chunks only.")
            with st.expander("Chunks used"):
                for ch in chunks:
                    st.caption(f"{ch['source']} · chunk {ch['chunk_idx'] + 1} · sim {ch['similarity']:.2f}")
                    st.text(ch["text"][:300])

        if not (a.error or b.error or c.error) and c.prompt_tokens:
            ratio = b.prompt_tokens / max(c.prompt_tokens, 1)
            per_day = (
                f"At 1,000 questions/day that's {b.prompt_tokens * 1000 / 1e6:.2f}M vs "
                f"{c.prompt_tokens * 1000 / 1e6:.2f}M input tokens per day."
            )
            if ratio >= 1.5:
                msg = (
                    f"**Prompt size:** the whole document used **{b.prompt_tokens:,}** tokens vs "
                    f"**{c.prompt_tokens:,}** for RAG (**{ratio:.0f}×** more). {per_day}"
                )
            else:
                msg = (
                    f"**Prompt size:** whole document **{b.prompt_tokens:,}** tokens vs RAG "
                    f"**{c.prompt_tokens:,}**. Your document is small enough that pasting it all "
                    "costs about the same (or less, since RAG adds its instructions). RAG pays off "
                    "once documents are much bigger than the top-k chunks. Try it with a long PDF."
                )
            st.info(msg, icon="💡")
        st.markdown("""
**What to look for**
- Did **A** invent an answer? That's hallucination: the model fills gaps with plausible text.
- Did **B** and **C** agree? If C missed something B found, retrieval picked the wrong chunks. Try a higher top-k.
- Which would you ship for 1,000 users a day, given the token numbers above?
""")

    st.divider()
    _fine_tuning_quiz()


# ── Fine-tuning: decide, don't run ────────────────────────────

_SCENARIOS = [
    {
        "s": "An HR assistant must answer from a 300-page policy handbook that changes every quarter, and quote the policy section.",
        "a": "RAG",
        "why": "The facts are large, they change, and answers need citations. Fine-tuning would go stale every quarter and can't cite sources.",
    },
    {
        "s": "A support bot must always reply in a strict JSON schema, 2 million times a day. A big model with a long prompt gets the format right, but it's too slow and expensive.",
        "a": "Fine-tuning",
        "why": "This is about behaviour and format, not facts, at high volume. Tuning a small model bakes the format in and cuts cost and latency. Measure it against the big model on an eval set first.",
    },
    {
        "s": "You want release notes written in your brand's friendly tone. You have 5 good examples and ship weekly.",
        "a": "Prompting",
        "why": "A few examples in the prompt (few-shot) handle the tone. The volume is low, so there's no reason to train anything.",
    },
    {
        "s": "Analysts ask one-off questions about a single 40-page contract they just uploaded.",
        "a": "Long context",
        "why": "One short document, read once. Pasting it all in is simplest and avoids retrieval misses. RAG starts to pay off with many documents or many repeated queries.",
    },
]


def _fine_tuning_quiz() -> None:
    st.markdown("#### 🧭 Decision drill: which approach?")
    st.caption("You can't fine-tune on free tiers, but you can practise the decision. Pick one, then reveal.")
    options = ["Prompting", "RAG", "Fine-tuning", "Long context"]
    for i, sc in enumerate(_SCENARIOS, 1):
        with st.container(border=True):
            st.markdown(f"**{i}.** {sc['s']}")
            pick = st.radio("Your call", options, key=f"ch1_quiz_{i}", horizontal=True, index=None)
            if pick:
                if pick == sc["a"]:
                    st.success(f"✅ {sc['a']}: {sc['why']}")
                else:
                    st.warning(f"Better answer: **{sc['a']}**. {sc['why']}")


def interview() -> None:
    interview_card(
        question="RAG vs fine-tuning vs prompting: which would you choose, and why?",
        asked_by="Perplexity, Google DeepMind",
        structure=[
            "Pick one for a concrete use case. Don't answer in the abstract.",
            "Say what the knowledge is: size, how often it changes, whether answers need citations.",
            "Justify with cost, freshness, latency and how you'd evaluate it.",
            "Say how the approaches combine, and what you'd try first.",
            "End with the condition that would change your choice.",
        ],
        sample=(
            "For a product answering questions over a company's documents, I'd choose RAG. "
            "The knowledge is large and changes weekly, and users need citations to trust answers. "
            "I'd start with strong prompting plus RAG and measure groundedness and answer accuracy on "
            "a golden set of about 100 real questions. I would not fine-tune for facts: it bakes in a "
            "snapshot, can't cite, and every update means retraining and re-evaluating. Fine-tuning "
            "comes later, if we hit volume where a smaller tuned model gives the same quality on format "
            "and tone at a fraction of the cost and latency."
        ),
        change_my_mind=(
            "If the knowledge were small and stable enough to fit in the prompt, I'd skip RAG and use "
            "prompting or long context. If the main problem were behaviour (format or tone) at very "
            "high volume, rather than facts, I'd fine-tune a small model."
        ),
        pitfalls=[
            "Hedging (\"it depends\") without ever committing.",
            "Thinking fine-tuning is how you teach a model new facts.",
            "Not mentioning evaluation: how you'd know the choice worked.",
            "Ignoring cost and latency at scale.",
        ],
    )


CHAPTER = {
    "id": "ch1",
    "title": "Prompting vs RAG vs fine-tuning",
    "short": "Prompt / RAG / FT",
    "tagline": "Where should the knowledge live? Commit to a choice, and know what would change it.",
    "concept": concept,
    "lab": lab,
    "interview": interview,
}
