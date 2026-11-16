# ============================================================
# Chapter 7 — Classic ML literacy
#
# Lab A: metrics on an imbalanced problem (why accuracy lies)
# Lab B: rules vs classic ML vs LLM on the same ticket-routing task
# Lab C: drill — which approach fits?
# ============================================================

from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from learn.common import call_model, interview_card, model_picker
from learn.tickets import CATEGORIES, RULES, TICKETS

TEST_PER_CLASS = 5


def concept() -> None:
    st.markdown("""
Amazon and Apple interviews in particular expect **ML fluency**: the metrics, the basic pitfalls, and judgement about when *not* to use an LLM.

**The confusion matrix: everything starts here**

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | True positive (TP) | False negative (FN), a *miss* |
| **Actually negative** | False positive (FP), a *false alarm* | True negative (TN) |

- **Accuracy** = (TP + TN) / all. *Misleading when classes are imbalanced*: with 1% fraud, "never fraud" is 99% accurate and useless.
- **Precision** = TP / (TP + FP). *When we flag something, how often are we right?* Matters when false alarms are costly (blocking a good customer).
- **Recall** = TP / (TP + FN). *Of all real positives, how many did we catch?* Matters when misses are costly (fraud, cancer screening).
- **F1** = the harmonic mean of precision and recall. A single number when both matter.
- **Thresholds** trade precision for recall. Chapter 3 picks the threshold from the cost of each kind of error.
- **ROC-AUC / PR-AUC** summarise performance across all thresholds. PR-AUC is more informative when positives are rare.

**Pitfalls worth naming**
- **Overfitting:** great on training data, poor on new data. Always evaluate on a held-out **test set**.
- **Data leakage:** a feature that secretly contains the answer, such as "refund issued" when predicting complaints.
- **Distribution shift:** the world changes after training. Monitor it in production.
- **Class imbalance:** use precision/recall or PR-AUC, not accuracy.

**Rules vs classic ML vs LLMs**

| | Rules | Classic ML (e.g. logistic regression, gradient boosting) | LLM |
|---|---|---|---|
| Needs | Domain knowledge | Labelled examples (hundreds+) | A prompt, maybe a few examples |
| Cost per prediction | ~0 | ~0 | Highest |
| Latency | Microseconds | Milliseconds | Seconds |
| Explainable | Fully | Partly (feature weights) | Weakly |
| Handles messy language | Poorly | OK with enough data | Very well |
| Best for | Clear, stable, auditable logic | High volume, structured or tabular data, with labels | Unstructured text, no labels yet, fast iteration |

A common real-world path: **launch with an LLM** (no labels needed), **collect its predictions plus human corrections as labels**, then **train a cheap classic model** for the high-volume path and keep the LLM for the hard cases.
""")


# ── Lab A: imbalance ──────────────────────────────────────────

def _lab_imbalance() -> None:
    st.markdown("A fraud model scores **2,000 transactions**. Fraud is rare. Watch what accuracy does.")
    c1, c2 = st.columns(2)
    with c1:
        rate = st.slider("Share of transactions that are fraud", 0.5, 30.0, 2.0, 0.5, key="ch7_rate") / 100
    with c2:
        t = st.slider("Flag as fraud if score ≥", 0.0, 1.0, 0.5, 0.01, key="ch7_t")

    rng = np.random.default_rng(3)
    n = 2000
    y = rng.random(n) < rate
    score = np.where(y, rng.beta(4, 2, n), rng.beta(1.5, 5, n))
    pred = score >= t
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    base_acc = 1 - y.mean()

    m = st.columns(5)
    m[0].metric("Accuracy", f"{acc:.1%}")
    m[1].metric("Precision", f"{prec:.0%}")
    m[2].metric("Recall", f"{rec:.0%}")
    m[3].metric("F1", f"{f1:.2f}")
    m[4].metric("'Never fraud' accuracy", f"{base_acc:.1%}", help="A model that flags nothing.")

    st.markdown(
        f"| | Flagged | Not flagged |\n|---|---|---|\n"
        f"| **Fraud** ({int(y.sum())}) | ✅ {tp} caught | ❌ {fn} missed |\n"
        f"| **Legit** ({int((~y).sum())}) | ⚠️ {fp} false alarms | ✅ {tn} |"
    )
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=score[~y], name="Legit", marker_color="#6C8EF5", opacity=0.65, nbinsx=40))
    fig.add_trace(go.Histogram(x=score[y], name="Fraud", marker_color="#ff6b6b", opacity=0.75, nbinsx=40))
    fig.add_vline(x=t, line_dash="dash", annotation_text="threshold")
    fig.update_layout(barmode="overlay", height=260, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Model score", yaxis_title="Transactions", legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"**Takeaway:** at {rate:.1%} fraud, a model that never flags anything scores **{base_acc:.1%} accuracy** "
        "and catches nothing. That's why imbalanced problems are judged on **precision and recall**, not accuracy."
    )


# ── Lab B: rules vs ML vs LLM ─────────────────────────────────

def _split(per_class_train: int):
    rng = np.random.default_rng(11)
    train, test = [], []
    for cat in CATEGORIES:
        rows = [t for t in TICKETS if t[1] == cat]
        idx = rng.permutation(len(rows))
        test += [rows[i] for i in idx[:TEST_PER_CLASS]]
        train += [rows[i] for i in idx[TEST_PER_CLASS:TEST_PER_CLASS + per_class_train]]
    return train, test


def _rules_predict(text: str) -> str:
    low = text.lower()
    for cat, words in RULES.items():
        if any(w in low for w in words):
            return cat
    return "(no rule matched)"


def _ml_predict(train, test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        LogisticRegression(max_iter=2000, C=5.0),
    )
    model.fit([t for t, _ in train], [c for _, c in train])
    return list(model.predict([t for t, _ in test])), model


def _top_words(model, k: int = 4) -> dict[str, list[str]]:
    vec, clf = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
    vocab = np.array(vec.get_feature_names_out())
    return {cls: list(vocab[np.argsort(coef)[-k:][::-1]]) for cls, coef in zip(clf.classes_, clf.coef_)}


def _llm_predict(rag, model: str, test) -> tuple[list[str], object]:
    numbered = "\n".join(f"{i}. {t}" for i, (t, _) in enumerate(test, 1))
    prompt = (
        f"Classify each customer-support ticket into exactly one category: {', '.join(CATEGORIES)}.\n\n"
        f"{numbered}\n\n"
        f"Reply with only a JSON array of {len(test)} category strings, in order."
    )
    res = call_model(rag, model, "You are a precise classifier. Output JSON only.", prompt)
    labels: list[str] = []
    m = re.search(r"\[.*\]", res.response or "", flags=re.S)
    if m:
        try:
            labels = [str(x).strip().lower() for x in json.loads(m.group(0))]
        except json.JSONDecodeError:
            labels = []
    labels = (labels + ["(no answer)"] * len(test))[: len(test)]
    return labels, res


def _score(y_true, y_pred) -> tuple[float, float]:
    from sklearn.metrics import f1_score
    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    f1 = float(f1_score(y_true, y_pred, labels=CATEGORIES, average="macro", zero_division=0))
    return acc, f1


def _lab_compare(rag) -> None:
    st.markdown(
        f"Route **{len(CATEGORIES) * TEST_PER_CLASS} held-out support tickets** into "
        f"{', '.join(CATEGORIES)} three ways. About a third of the tickets avoid the obvious "
        "keywords (e.g. *\"Where is my stuff?\"*)."
    )
    c1, c2 = st.columns(2)
    with c1:
        per_class = st.slider("Training examples per category (classic ML)", 2, 11, 11, key="ch7_ntrain",
                              help="Slide down to see how classic ML depends on labelled data.")
    with c2:
        model = model_picker("LLM", key="ch7_model", prefer="nvidia/nemotron-3.5-lightning:free")
    st.caption("Rules and classic ML run locally (free, instant). The LLM classifies all tickets in **1 batched call**.")

    train, test = _split(per_class)
    if st.button("▶️ Run all three", type="primary", key="ch7_go", disabled=not model):
        y = [c for _, c in test]
        t0 = time.perf_counter(); rules = [_rules_predict(t) for t, _ in test]; t_rules = time.perf_counter() - t0
        t0 = time.perf_counter(); ml, ml_model = _ml_predict(train, test); t_ml = time.perf_counter() - t0
        with st.spinner("LLM classifying…"):
            llm, res = _llm_predict(rag, model, test)
        st.session_state.ch7 = dict(test=test, y=y, rules=rules, ml=ml, llm=llm, res=res,
                                    t_rules=t_rules, t_ml=t_ml, top=_top_words(ml_model), n=per_class)

    r = st.session_state.get("ch7")
    if not r:
        return
    if r["res"].error:
        st.error(f"LLM call failed: {r['res'].error}")

    rows = []
    for name, preds, secs, explain in [
        ("Rules", r["rules"], r["t_rules"], "Fully: you can read the keyword that fired"),
        (f"Classic ML ({r['n']}/category)", r["ml"], r["t_ml"], "Partly: top-weighted words per class"),
        ("LLM", r["llm"], r["res"].wall_clock_s, "Weakly: you can ask for a reason, but it may not be faithful"),
    ]:
        acc, f1 = _score(r["y"], preds)
        rows.append({"Approach": name, "Accuracy": f"{acc:.0%}", "Macro F1": f"{f1:.2f}",
                     "Time": f"{secs * 1000:.0f} ms" if secs < 1 else f"{secs:.1f} s",
                     "Explainability": explain})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    mark = lambda p, t: ("✅ " if p == t else "❌ ") + p
    detail = pd.DataFrame({
        "Ticket": [t for t, _ in r["test"]],
        "True": r["y"],
        "Rules": [mark(p, t) for p, t in zip(r["rules"], r["y"])],
        "Classic ML": [mark(p, t) for p, t in zip(r["ml"], r["y"])],
        "LLM": [mark(p, t) for p, t in zip(r["llm"], r["y"])],
    })
    st.dataframe(detail, use_container_width=True, hide_index=True, height=420)
    with st.expander("What classic ML learned (top-weighted words per category)"):
        st.markdown("  \n".join(f"**{c}:** {', '.join(w)}" for c, w in r["top"].items()))
    st.markdown("""
**What to look for**
- Which tickets did **rules** miss? Usually the paraphrases with no keyword.
- **Classic ML is weak here**, because it learned from only 11 examples per category. Real text classifiers train on thousands. Slide down to 2 and it gets worse still. The LLM needed no labels at all. That's the real trade-off: labels vs cost per prediction.
- The LLM is likely most accurate here but by far the slowest and costliest **per prediction**. At 10 million tickets a month, that matters.
- With only 20 test tickets, one ticket is 5 points of accuracy. Small test sets are noisy.
""")


# ── Lab C: drill ──────────────────────────────────────────────

_DRILL = [
    ("Block card transactions over the customer's set spending limit.", "Rules",
     "The logic is exact, stable and must be auditable. No learning needed."),
    ("Predict which of 5M subscribers will churn next month, from usage and billing tables.", "Classic ML",
     "Tabular data, huge volume, years of labelled history. Gradient boosting is cheap, fast and strong here."),
    ("Sort free-text app reviews into themes for a new product, with no labelled data yet.", "LLM",
     "Unstructured language and no labels. An LLM works on day one, and its outputs can become labels later."),
    ("Flag likely-fraudulent insurance claims for a human investigator.", "Classic ML",
     "Structured claim features with labelled fraud history. Tune for recall and let humans review. An LLM could summarise claim notes as an extra feature."),
]


def _lab_drill() -> None:
    st.markdown("Pick the simplest approach that works.")
    opts = ["Rules", "Classic ML", "LLM"]
    for i, (s, a, why) in enumerate(_DRILL, 1):
        with st.container(border=True):
            st.markdown(f"**{i}.** {s}")
            pick = st.radio("Your call", opts, key=f"ch7_drill_{i}", horizontal=True, index=None)
            if pick:
                (st.success if pick == a else st.warning)(("✅ " if pick == a else f"Better: **{a}**. ") + why)


def lab(rag) -> None:
    ta, tb, tc = st.tabs(["A · Why accuracy lies", "B · Rules vs ML vs LLM", "C · Which approach?"])
    with ta:
        _lab_imbalance()
    with tb:
        _lab_compare(rag)
    with tc:
        _lab_drill()


def interview() -> None:
    interview_card(
        question="When would you use a rule-based system, a classic ML model, or an LLM?",
        asked_by="Amazon, Apple",
        structure=[
            "Frame the decision on four axes: data available (labels?), input type (structured vs free text), volume and cost, explainability and risk.",
            "Rules: clear, stable, auditable logic.",
            "Classic ML: lots of labelled, often tabular data at high volume.",
            "LLM: unstructured language, little or no labelled data, fast iteration.",
            "Show the hybrid path: LLM first, harvest labels, distil to a cheap model, keep the LLM for hard cases.",
            "Name the metric you'd judge by (precision/recall at the business threshold), not accuracy.",
        ],
        sample=(
            "I choose by the data and the cost of being wrong. If the logic is crisp and must be "
            "auditable, like spending limits, I use rules. If I have lots of labelled, structured data "
            "at high volume, like churn prediction on billing tables, classic ML such as gradient "
            "boosting wins on cost, latency and explainability. If the input is messy language and I "
            "have no labels, I'd start with an LLM. For example, I'd route support tickets with an "
            "LLM on day one, log its predictions plus agent corrections, and once there are a few "
            "thousand labels, train a small classifier for the 80% of easy tickets. Either way I'd "
            "evaluate on a held-out set using precision and recall per class, because ticket "
            "categories are imbalanced."
        ),
        change_my_mind=(
            "If explainability were legally required (credit decisions, for example), I'd avoid an LLM "
            "as the decision-maker even if it scored higher, and use an interpretable model with "
            "reason codes."
        ),
        pitfalls=[
            "Reaching for an LLM by default.",
            "Quoting accuracy on an imbalanced problem.",
            "Forgetting that classic ML needs labelled data, and where it comes from.",
            "Ignoring cost per prediction at scale.",
        ],
    )
    st.write("")
    interview_card(
        question="Explain precision and recall to a non-technical stakeholder, and tell me which matters more for a fraud model.",
        asked_by="Amazon",
        structure=[
            "Use a concrete analogy: a fishing net, or an airport scanner.",
            "Precision: of the things we flagged, how many were really fraud.",
            "Recall: of all the real fraud, how much we caught.",
            "Tie the choice to costs: a missed fraud vs a blocked good customer.",
            "Commit to a choice, and say how you'd pick the threshold.",
        ],
        sample=(
            "Think of an airport scanner. Precision is: when it beeps, how often is there really "
            "something dangerous? Recall is: of all the dangerous items, how many did it catch? For "
            "fraud I'd prioritise recall at the first stage, because a missed fraud costs the full "
            "amount plus chargeback fees. But I'd protect precision by sending flagged transactions to "
            "step-up verification or human review, not outright blocking, so a false alarm costs a few "
            "seconds of customer friction rather than a lost customer. I'd set the threshold where the "
            "expected cost of the two errors is lowest."
        ),
        change_my_mind=(
            "If every flag meant a hard block with no review, false alarms would directly lose good "
            "customers, and I'd shift toward precision."
        ),
        pitfalls=[
            "Defining the terms without a concrete example.",
            "Saying \"both matter\" without choosing.",
            "Not connecting the choice to business costs.",
        ],
    )


CHAPTER = {
    "id": "ch7",
    "title": "Classic ML literacy",
    "short": "ML literacy",
    "tagline": "Precision, recall and pitfalls, and when rules or classic ML beat an LLM.",
    "concept": concept,
    "lab": lab,
    "interview": interview,
}
