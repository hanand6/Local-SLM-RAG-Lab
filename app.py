# ============================================================
# app.py — Local SLM RAG App (Streamlit)
#
# Tabs:
#   1. Setup & Status     — Ollama health, model availability
#   2. Upload & Index     — Ingest documents locally
#   3. Ask a Question     — Single-model RAG Q&A with streaming
#   4. Compare Models     — Same question → 4 models side-by-side
#   5. Benchmark Suite    — Structured perf tests across models
#   6. Tradeoff Report    — Quality vs Speed vs Cost analysis
# ============================================================

import os
import time
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    MODELS, BENCHMARK_PROMPTS, OLLAMA_BASE_URL, NVIDIA_API_KEY, GEMINI_API_KEY,
    OPENROUTER_API_KEY, EMBEDDING_MODEL,
)
from api_client import list_openrouter_free_models
from learn.ui import render_learn
from embedder import EMBEDDING_OPTIONS
from ollama_client import OllamaClient
from rag_engine import RAGEngine
from comparator import ModelComparator
from benchmarker import Benchmarker, BenchmarkResult
from ragas_evaluator import RagasEvaluator, JUDGE_OPTIONS


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert #RRGGBB hex to rgba() string. Plotly requires this for fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Page config ──────────────────────────────────────────────

st.set_page_config(
    page_title="SLM RAG Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;500;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
  }
  code, pre, .stCode { font-family: 'JetBrains Mono', monospace; }

  .metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #e0e0e0;
  }
  .metric-card h4 { color: #a0c4ff; margin: 0 0 4px 0; font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .metric-card .val { font-size: 1.6rem; font-weight: 700; color: #ffffff; }
  .metric-card .sub { font-size: 0.75rem; color: #8892b0; margin-top: 2px; }

  .badge-local  { background: #1a4731; color: #3effa3; border-radius: 6px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }
  .badge-warn   { background: #4a2f00; color: #ffb347; border-radius: 6px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }
  .badge-error  { background: #4a0000; color: #ff6b6b; border-radius: 6px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }

  .answer-box {
    background: #0d1117;
    border-left: 3px solid #3effa3;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    color: #cdd9e5;
    font-size: 0.93rem;
    line-height: 1.6;
    margin-top: 8px;
  }
  .chunk-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.8rem;
    font-size: 0.82rem;
    color: #8892b0;
    margin-top: 4px;
  }

  .stTabs [data-baseweb="tab"] { font-size: 0.9rem; font-weight: 500; }
  .stTabs [aria-selected="true"] { color: #3effa3 !important; }

  h1 { font-size: 1.8rem !important; }
  h2 { font-size: 1.3rem !important; color: #a0c4ff; }
  h3 { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ───────────────────────────────────────

def init_state():
    # Indexing config — stored separately so we can detect changes
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = EMBEDDING_MODEL
    if "similarity" not in st.session_state:
        st.session_state.similarity = "cosine"
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 500
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 60

    if "rag" not in st.session_state:
        st.session_state.rag = RAGEngine(
            embedding_model=st.session_state.embed_model,
            similarity=st.session_state.similarity,
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
        )
    if "comparator" not in st.session_state:
        st.session_state.comparator = ModelComparator(st.session_state.rag)
    if "benchmark_results" not in st.session_state:
        st.session_state.benchmark_results = []
    if "compare_results" not in st.session_state:
        st.session_state.compare_results = []
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = list(MODELS.keys())
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None

init_state()
client = OllamaClient()
rag: RAGEngine = st.session_state.rag
comparator: ModelComparator = st.session_state.comparator


# ── Sidebar ──────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 SLM RAG Lab")
    st.caption("Free OpenRouter models by default. Local Ollama models optional.")

    st.divider()

    # OpenRouter key status
    if OPENROUTER_API_KEY:
        st.markdown("**OpenRouter:** 🟢 Key set")
    else:
        st.markdown("**OpenRouter:** 🔴 No key")
        st.error(
            "Add `OPENROUTER_API_KEY=...` to `.env` "
            "(free key: [openrouter.ai/keys](https://openrouter.ai/keys)), then restart.",
            icon="🔑",
        )

    # Ollama status (optional)
    ollama_ok = client.is_running()
    ollama_models = client.list_models() if ollama_ok else []
    st.markdown(
        f"**Ollama (optional):** {'🟢 Running' if ollama_ok else '⚪ Not running'}"
    )

    # Index status
    chunk_count = rag.chunk_count
    doc_count = len(rag.indexed_docs)
    st.markdown(f"**Indexed:** {doc_count} doc(s) · {chunk_count} chunks")

    st.divider()

    # Model selection
    st.markdown("### Models in scope")
    selected = []
    for model_id, meta in MODELS.items():
        backend = meta.get("backend", "ollama")
        is_cloud = not meta.get("data_local", True)

        if backend == "ollama":
            available = any(model_id in m or m.startswith(model_id) for m in ollama_models)
            status_icon = "✅" if available else "⬜"
            default_on = available
        else:
            # Cloud models are always reachable (key check happens at query time)
            status_icon = "☁️"
            # Default: only the free OpenRouter models are ticked
            default_on = backend == "openrouter"

        label = f"{status_icon} {meta['label']}"
        checked = st.checkbox(label, value=default_on, key=f"chk_{model_id}")
        if checked:
            selected.append(model_id)

    st.session_state.selected_models = selected

    # Cloud privacy warning
    cloud_selected = [m for m in selected if not MODELS[m].get("data_local", True)]
    if cloud_selected:
        st.warning(
            "⚠️ Cloud models selected. Documents sent to these models will "
            "**leave your device**. Do not use with confidential data.",
            icon="🔓"
        )

    st.divider()
    st.markdown("### 🔒 Privacy Status")
    if cloud_selected:
        st.markdown(
            "- **Embeddings & vector store:** local\n"
            "- **Generation:** cloud API — retrieved chunks leave your device\n"
            "- **Use:** non-sensitive documents only"
        )
    else:
        st.markdown(
            "- **Embeddings & vector store:** local\n"
            "- **Generation:** local Ollama only\n"
            "- **Data leaves device:** No"
        )


# ── Main area ────────────────────────────────────────────────

st.title("🧠 SLM RAG Lab")
st.caption("Document Q&A · Model Comparison · Benchmarking · Evaluation")

tab1, tab2, tab3, tab_learn, tab4, tab5, tab6, tab7 = st.tabs([
    "⚙️ Setup",
    "📄 Upload & Index",
    "💬 Ask",
    "🎓 Learn",
    "⚖️ Compare",
    "📊 Benchmark",
    "📈 Tradeoff Report",
    "🧪 Eval (RAGAS)",
])


# ════════════════════════════════════════════════════════════
# TAB 1 — SETUP & STATUS
# ════════════════════════════════════════════════════════════
with tab1:
    st.header("System Setup & Status")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🌐 OpenRouter (default)")
        if OPENROUTER_API_KEY:
            st.success("OPENROUTER_API_KEY is set — free models ready")
        else:
            st.error("OPENROUTER_API_KEY not set")
            st.markdown(
                "1. Create a free key at [openrouter.ai/keys](https://openrouter.ai/keys)\n"
                "2. Add `OPENROUTER_API_KEY=sk-or-...` to `.env`\n"
                "3. Restart the app"
            )
        st.caption(
            "Free models are rate-limited (roughly 20 requests/min plus a daily cap). "
            "Rate-limited calls are retried automatically with backoff."
        )

    with col2:
        st.subheader("Free models available right now")

        @st.cache_data(ttl=3600, show_spinner=False)
        def _free_models():
            return list_openrouter_free_models()

        try:
            free = _free_models()
            configured = {m for m, meta in MODELS.items() if meta.get("backend") == "openrouter"}
            st.dataframe(
                pd.DataFrame([
                    {
                        "In use": "✅" if m["id"] in configured else "",
                        "Model ID": m["id"],
                        "Context (k)": m["context_length"] // 1000,
                    }
                    for m in free
                ]),
                use_container_width=True, hide_index=True, height=240,
            )
            missing = configured - {m["id"] for m in free}
            if missing:
                st.warning(
                    "No longer free/available: " + ", ".join(sorted(missing))
                    + ". Update `OPENROUTER_MODELS` in `.env`."
                )
            st.caption(
                "To swap models, set `OPENROUTER_MODELS=id1,id2,...` in `.env` and restart."
            )
        except Exception as e:
            st.info(f"Could not fetch the OpenRouter model list: {e}")

    with st.expander("🖥️ Optional: local models with Ollama"):
        if ollama_ok:
            st.success(f"Ollama is running on {OLLAMA_BASE_URL}")
            st.write(f"**Pulled models ({len(ollama_models)}):** "
                     + (", ".join(f"`{m}`" for m in ollama_models) or "none"))
        else:
            st.info("Ollama is not running. That's fine — it's only needed for local models.")
        pull_cmds = "\n".join(
            f"{meta['pull_cmd']}  # {meta['label']} — {meta['disk_size_gb']}GB"
            for mid, meta in MODELS.items() if meta.get("backend") == "ollama"
        )
        pull_cmds += "\nollama pull nomic-embed-text  # optional embedding model"
        st.code("ollama serve\n" + pull_cmds, language="bash")

    st.divider()
    st.subheader("Model Specifications")
    specs = []
    for mid, meta in MODELS.items():
        backend = meta.get("backend", "ollama")
        is_cloud = not meta.get("data_local", True)
        if backend == "ollama":
            available = any(mid in m or m.startswith(mid) for m in ollama_models)
            status = "✅ Available" if available else "⬜ Not pulled"
        else:
            # Check if API key is configured
            has_key = {
                "openrouter": bool(OPENROUTER_API_KEY),
                "nvidia_nim": bool(NVIDIA_API_KEY),
                "gemini":     bool(GEMINI_API_KEY),
            }.get(backend, False)
            status = "🔑 Key set" if has_key else "🔴 No API key"
        specs.append({
            "Model":        meta["label"],
            "Family":       meta["family"],
            "Params (B)":   meta["params_b"] if meta["params_b"] else "N/A",
            "Context (k)":  f"{meta['context_len']//1000}k" if meta["context_len"] else "?",
            "Disk (GB)":    meta["disk_size_gb"] if not is_cloud else "☁️ Cloud",
            "Backend":      "☁️ API" if is_cloud else "🖥️ Local",
            "Data private": "✅ Yes" if not is_cloud else "❌ No — cloud",
            "Status":       status,
        })
    st.dataframe(pd.DataFrame(specs), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🔒 Privacy & Cost Guarantees")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**Privacy (Local models)**
- ✅ Zero network egress
- ✅ Documents stay on device
- ✅ No API keys needed
- ✅ GDPR / HIPAA friendly
- ✅ ChromaDB telemetry disabled
        """)
    with c2:
        st.markdown("""
**Cloud models (OpenRouter, Kimi K2, Gemini)**
- ❌ Retrieved chunks sent to remote servers
- ⚠️ Free tiers may log prompts — not for confidential docs
- 🔑 API key required
- ⏱️ Rate limits on free tiers
- ✅ No local GPU or model downloads needed
        """)
    with c3:
        st.markdown("""
**API Key Setup (in `.env`)**
- `OPENROUTER_API_KEY` — default ([openrouter.ai/keys](https://openrouter.ai/keys))
- `NVIDIA_API_KEY` — optional ([build.nvidia.com](https://build.nvidia.com))
- `GEMINI_API_KEY` — optional ([aistudio.google.com](https://aistudio.google.com))
        """)

    st.divider()
    st.subheader("🔑 API Key Status")
    k0, k1, k2 = st.columns(3)
    with k0:
        if OPENROUTER_API_KEY:
            st.success("✅ OPENROUTER_API_KEY is set")
        else:
            st.error("❌ OPENROUTER_API_KEY not set (required for default models)")
    with k1:
        if NVIDIA_API_KEY:
            st.success("✅ NVIDIA_API_KEY is set — Kimi K2 ready")
        else:
            st.error("❌ NVIDIA_API_KEY not set — set env var to use Kimi K2")
    with k2:
        if GEMINI_API_KEY:
            st.success("✅ GEMINI_API_KEY is set — Gemini 3 ready")
        else:
            st.error("❌ GEMINI_API_KEY not set — set env var to use Gemini 3")


# ════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD & INDEX
# ════════════════════════════════════════════════════════════
with tab2:
    st.header("Upload & Index Documents")
    st.caption("Supported: PDF, DOCX, TXT, MD — all processed locally, no uploads to any server.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Step 1 — Choose files**")
        uploaded = st.file_uploader(
            "Drop documents here",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
        )

        indexed_names = {d["doc_name"] for d in rag.indexed_docs}
        if uploaded:
            pending = [uf for uf in uploaded if uf.name not in indexed_names]
            already = [uf.name for uf in uploaded if uf.name in indexed_names]

            st.markdown("**Step 2 — Index them**")
            for uf in uploaded:
                size_mb = uf.size / 1_000_000
                state = "✅ indexed" if uf.name in indexed_names else "⏳ ready to index"
                st.markdown(f"📄 `{uf.name}` · {size_mb:.1f} MB · {state}")

            if pending:
                label = (
                    f"📥 Index {pending[0].name}" if len(pending) == 1
                    else f"📥 Index {len(pending)} documents"
                )
                if st.button(label, type="primary", use_container_width=True, key="index_uploads"):
                    ok, failed = [], []
                    for i, uf in enumerate(pending, start=1):
                        bar = st.progress(0.0, text=f"[{i}/{len(pending)}] {uf.name}: reading and chunking...")

                        def _progress(done, total, _bar=bar, _name=uf.name, _i=i, _t0=time.perf_counter()):
                            elapsed = time.perf_counter() - _t0
                            eta = elapsed / done * (total - done) if done else 0
                            _bar.progress(
                                done / total,
                                text=f"[{_i}/{len(pending)}] {_name}: embedded {done:,}/{total:,} chunks · ~{eta:.0f}s left",
                            )

                        # Save under the original name so citations show it.
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            tmp_path = Path(tmp_dir) / Path(uf.name).name
                            tmp_path.write_bytes(uf.getbuffer())
                            try:
                                stats = rag.index_file(str(tmp_path), progress_cb=_progress)
                                ok.append(stats)
                            except Exception as e:
                                failed.append((uf.name, str(e)))
                        bar.empty()

                    st.session_state.index_flash = {"ok": ok, "failed": failed}
                    st.rerun()
            elif already:
                st.caption("All selected files are already indexed.")

        # ── Result of the last indexing run + next step ─────────
        flash = st.session_state.pop("index_flash", None)
        if flash:
            for s in flash["ok"]:
                st.success(
                    f"✅ **{s['doc_name']}** — {s['chunks']:,} chunks · "
                    f"{s['char_count']:,} chars · {s['index_time_s']}s"
                )
            for name, err in flash["failed"]:
                st.error(f"❌ **{name}** — {err}")
            if not flash["ok"] and not flash["failed"]:
                st.warning("Nothing was indexed.")
            if flash["ok"] and any(s["chunks"] == 0 for s in flash["ok"]):
                st.warning(
                    "Some documents produced 0 chunks — likely a scanned PDF with no "
                    "text layer. Those can't be searched."
                )

        if rag.chunk_count > 0:
            st.info(
                f"**You're ready for the next step.** {len(rag.indexed_docs)} document(s) · "
                f"{rag.chunk_count:,} chunks indexed.\n\n"
                "➡️ Open the **💬 Ask** tab above to ask a question, "
                "**🎓 Learn** for guided chapters, or **⚖️ Compare** to try several models side by side.",
                icon="🎉",
            )

        st.divider()
        st.subheader("Or paste text directly")
        pasted = st.text_area("Paste document content here", height=150)
        paste_name = st.text_input("Document name (for citation)", value="pasted_doc")
        if st.button("📥 Index pasted text", disabled=not pasted.strip()):
            with st.spinner("Indexing..."):
                try:
                    stats = rag.index_text(pasted, doc_name=paste_name)
                    st.session_state.index_flash = {"ok": [stats], "failed": []}
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.subheader("⚙️ Indexing Configuration")

        # ── Embedding model ────────────────────────────────────
        embed_options = list(EMBEDDING_OPTIONS.keys())
        new_embed = st.selectbox(
            "Embedding model",
            options=embed_options,
            index=embed_options.index(st.session_state.embed_model),
            format_func=lambda m: EMBEDDING_OPTIONS[m]["label"],
            key="embed_select",
        )

        # ── Similarity metric ──────────────────────────────────
        sim_options = ["cosine", "l2"]
        sim_labels  = {"cosine": "Cosine (default)", "l2": "Squared L2 (Euclidean)"}
        new_sim = st.selectbox(
            "Similarity metric",
            options=sim_options,
            index=sim_options.index(st.session_state.similarity),
            format_func=lambda s: sim_labels[s],
            key="sim_select",
        )

        # ── Warn + apply if embed/similarity changed ──────────
        needs_reset = (
            new_embed != rag.embedding_model
            or new_sim != rag.similarity
        )
        if needs_reset:
            st.warning(
                "⚠️ Changing embedding model or similarity metric requires "
                "**resetting the index** (different vector space). "
                "All indexed documents will be cleared.",
            )
            if st.button("🔄 Apply & Reset Index", type="primary", key="apply_embed"):
                changed = rag.reconfigure_embedder(new_embed, new_sim)
                if changed:
                    st.session_state.embed_model = new_embed
                    st.session_state.similarity  = new_sim
                    st.success(
                        f"✅ Switched to **{EMBEDDING_OPTIONS[new_embed]['label']}** · "
                        f"**{sim_labels[new_sim]}**. Index cleared."
                    )
                    st.rerun()

        st.divider()

        # ── Chunk size & overlap sliders ───────────────────────
        new_chunk_size = st.slider(
            "Chunk size (chars)",
            min_value=100, max_value=2000,
            value=st.session_state.chunk_size,
            step=50, key="chunk_size_slider",
            help="Characters per chunk. Smaller = more precise retrieval. Larger = more context per chunk.",
        )
        new_overlap = st.slider(
            "Chunk overlap (chars)",
            min_value=0, max_value=400,
            value=st.session_state.chunk_overlap,
            step=10, key="overlap_slider",
            help="Characters of overlap between adjacent chunks to preserve context at boundaries.",
        )

        if new_chunk_size != st.session_state.chunk_size or new_overlap != st.session_state.chunk_overlap:
            rag.set_chunker_params(new_chunk_size, new_overlap)
            st.session_state.chunk_size    = new_chunk_size
            st.session_state.chunk_overlap = new_overlap

        st.divider()
        st.subheader("Index Status")
        st.metric("Total chunks", rag.chunk_count)
        st.metric("Documents", len(rag.indexed_docs))

        if rag.indexed_docs:
            st.write("**Indexed files:**")
            for doc in rag.indexed_docs:
                st.markdown(f"""
<div class="chunk-box">
  📄 <b>{doc['doc_name']}</b><br>
  {doc['chunks']} chunks · {doc['avg_chunk_len']} avg chars · {doc['index_time_s']}s
</div>
                """, unsafe_allow_html=True)

        st.divider()
        if st.button("🗑️ Clear Index", type="secondary", use_container_width=True):
            rag.reset()
            st.success("Index cleared — ready for a new document.")
            st.rerun()

        st.divider()
        # ── Live config summary ─────────────────────────────────
        st.subheader("Active Config")
        st.markdown(f"""
| Setting | Value |
|---|---|
| Chunk size | {rag.chunker.chunk_size} chars |
| Overlap | {rag.chunker.overlap} chars |
| Embedding model | `{rag.embedding_model}` |
| Embedding dim | {EMBEDDING_OPTIONS[rag.embedding_model]['dim']} |
| Similarity | {sim_labels.get(rag.similarity, rag.similarity)} |
| Vector store | ChromaDB (local) |
        """)


# ════════════════════════════════════════════════════════════
# LEARN — chapters with concept, lab and interview card
# ════════════════════════════════════════════════════════════
with tab_learn:
    render_learn(rag)


# ════════════════════════════════════════════════════════════
# TAB 3 — ASK A QUESTION
# ════════════════════════════════════════════════════════════
with tab3:
    st.header("Ask a Question")

    c1, c2 = st.columns([3, 1])
    with c1:
        question = st.text_input(
            "Your question",
            placeholder="e.g. What are the key findings in this document?",
        )
    with c2:
        ask_model = st.selectbox(
            "Model",
            options=st.session_state.selected_models,
            format_func=lambda m: MODELS[m]["label"],
        )

    top_k = st.slider("Retrieved chunks (top-k)", 2, 10, 5)
    use_bm25 = st.toggle(
        "🔀 BM25 Reranker — show results with AND without reranking side by side",
        value=False,
        help="Fetches 2× top-k from vector search, then BM25 reranks. Runs LLM twice.",
    )
    if use_bm25:
        st.caption(
            "⚠️ BM25 mode runs **two LLM calls** — one per chunk ordering. "
            "This takes roughly 2× as long as a normal query."
        )

    if st.button("🔍 Ask", type="primary") and question.strip():
        if rag.chunk_count == 0:
            st.warning("No documents indexed yet. Go to **Upload & Index** first.")
        else:
            backend = MODELS.get(ask_model, {}).get("backend", "ollama")
            is_cloud = not MODELS.get(ask_model, {}).get("data_local", True)

            if is_cloud and rag.chunk_count > 0:
                st.warning(
                    "⚠️ **Cloud model selected.** Your retrieved document chunks will be "
                    "sent to an external API. Only proceed if you are comfortable with this.",
                    icon="🔓"
                )

            if backend == "ollama" and ask_model not in ollama_models:
                st.error(
                    f"Local model {ask_model} is not available. Start Ollama and run "
                    f"`ollama pull {ask_model}`, or pick an OpenRouter model."
                )

            elif use_bm25:
                # ── BM25 COMPARISON MODE ── run LLM twice
                with st.spinner(f"Running Vector + BM25 comparison with {MODELS[ask_model]['label']}..."):
                    vec_result, bm25_result = rag.query_compare(question, ask_model, top_k=top_k)

                st.subheader("Answers: Vector Search vs BM25 Reranked")
                col_v, col_b = st.columns(2)

                with col_v:
                    st.markdown("#### 🔍 Vector Search")
                    st.caption("Chunks ordered by embedding similarity")
                    st.markdown(
                        f'<div class="answer-box">{vec_result["answer"]}</div>',
                        unsafe_allow_html=True
                    )
                    st.caption(
                        f"⚡ {vec_result['tokens_per_sec']:.1f} t/s · "
                        f"🕒 {vec_result['total_ms']:.0f}ms total · "
                        f"TTFT {vec_result['ttft_ms']:.0f}ms"
                    )
                    with st.expander(f"📎 Vector chunks ({len(vec_result['context_chunks'])})"):
                        for i, c in enumerate(vec_result["context_chunks"]):
                            st.markdown(
                                f'<div class="chunk-box"><b>#{i+1}</b> · {c["source"]} · '
                                f'Sim: {round(c["similarity"]*100)}%<br><br>{c["text"][:350]}...</div>',
                                unsafe_allow_html=True
                            )

                with col_b:
                    st.markdown("#### 📊 BM25 Reranked")
                    st.caption("Chunks reordered by keyword relevance (BM25)")
                    st.markdown(
                        f'<div class="answer-box">{bm25_result["answer"]}</div>',
                        unsafe_allow_html=True
                    )
                    st.caption(
                        f"⚡ {bm25_result['tokens_per_sec']:.1f} t/s · "
                        f"🕒 {bm25_result['total_ms']:.0f}ms total · "
                        f"TTFT {bm25_result['ttft_ms']:.0f}ms"
                    )
                    with st.expander(f"📎 BM25 chunks ({len(bm25_result['context_chunks'])})"):
                        for i, c in enumerate(bm25_result["context_chunks"]):
                            vec_rank = c.get("vector_rank", "?")
                            bm25_score = c.get("bm25_score", 0)
                            st.markdown(
                                f'<div class="chunk-box"><b>BM25 #{i+1}</b> (was vector #{vec_rank}) · '
                                f'{c["source"]} · BM25: {bm25_score:.3f}<br><br>{c["text"][:350]}...</div>',
                                unsafe_allow_html=True
                            )

            else:
                # ── STANDARD MODE ──
                with st.spinner(f"Querying {MODELS[ask_model]['label']}..."):
                    result = rag.query(question, ask_model, top_k=top_k)

                # Answer
                st.subheader("Answer")
                st.markdown(
                    f'<div class="answer-box">{result["answer"]}</div>',
                    unsafe_allow_html=True
                )

                # Performance metrics
                st.subheader("Performance")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Retrieval", f"{result['retrieval_ms']:.0f} ms")
                m2.metric("TTFT", f"{result['ttft_ms']:.0f} ms")
                m3.metric("Generation", f"{result['generation_ms']:.0f} ms")
                m4.metric("Total", f"{result['total_ms']:.0f} ms")
                m5.metric("Tokens/sec", f"{result['tokens_per_sec']:.1f}")

                # Context chunks
                with st.expander(f"📎 Retrieved context ({len(result['context_chunks'])} chunks)"):
                    for i, chunk in enumerate(result["context_chunks"]):
                        sim_pct = round(chunk["similarity"] * 100)
                        st.markdown(
                            f'<div class="chunk-box">'
                            f'<b>Chunk {i+1}</b> · Source: {chunk["source"]} · '
                            f'Similarity: {sim_pct}%<br><br>{chunk["text"][:400]}...'
                            f'</div>',
                            unsafe_allow_html=True
                        )


# ════════════════════════════════════════════════════════════
# TAB 4 — COMPARE MODELS
# ════════════════════════════════════════════════════════════
with tab4:
    st.header("Compare Models Side-by-Side")
    st.caption("Same question, same retrieved context — only the LLM changes.")

    cq = st.text_input(
        "Comparison question",
        placeholder="e.g. Summarize the main argument of this document.",
        key="compare_q"
    )
    c_ctrl1, c_ctrl2 = st.columns([3, 1])
    with c_ctrl1:
        compare_top_k = st.slider("Top-k chunks", 2, 8, 5, key="compare_topk")
    with c_ctrl2:
        use_bm25_compare = st.toggle(
            "🔀 BM25 Reranker",
            value=False,
            key="bm25_compare",
            help="Runs each model twice — once with vector chunks, once with BM25 reranked chunks.",
        )
    models_to_compare = st.session_state.selected_models

    if use_bm25_compare:
        n_models = len(models_to_compare)
        st.caption(
            f"⚠️ BM25 mode: {n_models} model(s) × 2 chunk orderings = "
            f"{n_models * 2} LLM calls total."
        )

    if st.button("⚖️ Run Comparison", type="primary") and cq.strip():
        if rag.chunk_count == 0:
            st.warning("Index documents first.")
        elif not models_to_compare:
            st.warning("Select at least one model in the sidebar.")
        else:
            if use_bm25_compare:
                # Run query_compare for each model
                vec_results, bm25_results = [], []
                total = len(models_to_compare)
                prog = st.progress(0)
                stat_txt = st.empty()
                for i, model in enumerate(models_to_compare):
                    stat_txt.text(
                        f"[{i+1}/{total}] {MODELS[model]['label']} · Vector + BM25..."
                    )
                    vr, br = rag.query_compare(cq, model, top_k=compare_top_k)
                    vec_results.append(vr)
                    bm25_results.append(br)
                    prog.progress((i + 1) / total)
                prog.empty(); stat_txt.empty()
                st.session_state.compare_results = vec_results   # keep for Tradeoff tab
                st.session_state.compare_bm25_results = bm25_results
            else:
                progress = st.progress(0)
                status_txt = st.empty()

                def on_progress(model, idx, total):
                    progress.progress(idx / total)
                    status_txt.text(f"Querying {MODELS[model]['label']}... ({idx}/{total})")

                results = comparator.compare(
                    cq, models_to_compare, top_k=compare_top_k,
                    progress_cb=on_progress
                )
                progress.empty(); status_txt.empty()
                st.session_state.compare_results = results
                st.session_state.pop("compare_bm25_results", None)

    # ── Results display ───────────────────────────────────────────
    if st.session_state.compare_results:
        has_bm25 = bool(st.session_state.get("compare_bm25_results"))

        def _render_answer_cols(res_list, tag=""):
            """Render a row of model answer cards."""
            cols = st.columns(len(res_list))
            for col, r in zip(cols, res_list):
                label = MODELS.get(r["model"], {}).get("label", r["model"])
                color = MODELS.get(r["model"], {}).get("color", "#ffffff")
                with col:
                    st.markdown(
                        f"<h3 style='color:{color}'>{label}</h3>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div class="answer-box">{r["answer"]}</div>',
                        unsafe_allow_html=True
                    )
                    st.caption(
                        f"⚡ {r['tokens_per_sec']:.1f} t/s · "
                        f"🕒 {r['total_ms']:.0f}ms · "
                        f"TTFT {r['ttft_ms']:.0f}ms"
                    )

        st.subheader("Answers")
        if has_bm25:
            vtab, btab = st.tabs(["🔍 Vector Search", "📊 BM25 Reranked"])
            with vtab:
                st.caption("Chunks ordered by embedding similarity")
                _render_answer_cols(st.session_state.compare_results)
            with btab:
                st.caption("Chunks reordered by BM25 keyword relevance")
                _render_answer_cols(st.session_state.compare_bm25_results)
        else:
            _render_answer_cols(st.session_state.compare_results)

        st.divider()
        st.subheader("Metrics Comparison")
        metrics_df = comparator.to_metrics_df(st.session_state.compare_results)
        display_cols = ["Model", "TTFT (ms)", "Generation (ms)", "Total latency (ms)",
                        "Tokens/sec", "Output tokens", "Local cost $", "Privacy"]
        st.dataframe(
            metrics_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        # Speed bar chart
        fig = px.bar(
            metrics_df,
            x="Model",
            y="Tokens/sec",
            color="Model",
            color_discrete_map={
                MODELS[m]["label"]: MODELS[m]["color"] for m in MODELS
            },
            title="Generation Speed (Tokens/sec) — higher is better",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Latency bar chart
        fig2 = px.bar(
            metrics_df,
            x="Model",
            y="Total latency (ms)",
            color="Model",
            title="Total Latency (ms) — lower is better",
            template="plotly_dark",
        )
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 5 — BENCHMARK SUITE
# ════════════════════════════════════════════════════════════
with tab5:
    st.header("Inference Benchmark Suite")
    st.caption("Structured benchmark: each model runs the same prompt set. Captures P50/P95 latency, TPS, TTFT.")

    col1, col2 = st.columns(2)
    with col1:
        bench_models = st.multiselect(
            "Models to benchmark",
            options=list(MODELS.keys()),
            default=st.session_state.selected_models,
            format_func=lambda m: MODELS[m]["label"],
        )
    with col2:
        n_runs = st.slider("Runs per prompt (for P50/P95)", 1, 5, 2)

    b_ctrl1, b_ctrl2 = st.columns(2)
    with b_ctrl1:
        use_rag_prompts = st.toggle("Use RAG prompts (requires indexed docs)", value=rag.chunk_count > 0)
    with b_ctrl2:
        use_bm25_bench = st.toggle(
            "🔀 BM25 Reranker comparison",
            value=False,
            key="bm25_bench",
            help="Runs each prompt twice per model: vector retrieval AND BM25 reranked. Halves speed but shows reranker impact on latency.",
        )

    if use_rag_prompts and rag.chunk_count > 0:
        prompts = BENCHMARK_PROMPTS
        system = (
            "You are a document Q&A assistant. Answer concisely from the provided context."
        )
    else:
        prompts = [
            "What is machine learning? Explain in 3 sentences.",
            "List 5 benefits of running AI models locally.",
            "What are the differences between supervised and unsupervised learning?",
        ]
        system = ""

    multiplier = 2 if use_bm25_bench else 1
    total_calls_est = len(prompts) * len(bench_models) * n_runs * multiplier
    st.write(
        f"**{len(prompts)} prompts × {len(bench_models)} models × {n_runs} runs"
        + (" × 2 (Vector + BM25)" if use_bm25_bench else "")
        + f" = {total_calls_est} total LLM calls**"
    )

    if st.button("▶️ Run Benchmark Suite", type="primary") and bench_models:
        all_results: list[BenchmarkResult] = []
        bm = Benchmarker()
        total_calls = len(bench_models) * len(prompts) * n_runs * multiplier
        done = 0
        progress = st.progress(0)
        status = st.empty()

        for model in bench_models:
            for p_idx, prompt in enumerate(prompts):
                backend = MODELS.get(model, {}).get("backend", "ollama")

                # Build prompt variants: vector-only, and optionally BM25
                prompt_variants: list[tuple[str, str]] = []  # (label, full_prompt)
                if use_rag_prompts and rag.chunk_count > 0:
                    vec_chunks, bm25_chunks = rag.retrieve_reranked(prompt, top_k=5)
                    _, vec_prompt  = rag.build_prompt(prompt, vec_chunks)
                    prompt_variants.append(("[Vector]", vec_prompt))
                    if use_bm25_bench:
                        _, bm25_prompt = rag.build_prompt(prompt, bm25_chunks)
                        prompt_variants.append(("[BM25]", bm25_prompt))
                else:
                    prompt_variants.append(("", prompt))
                    if use_bm25_bench:
                        prompt_variants.append(("[BM25]", prompt))  # same prompt, tag only

                for variant_label, full_prompt in prompt_variants:
                    for run in range(n_runs):
                        done += 1
                        progress.progress(done / total_calls)
                        status.text(
                            f"[{done}/{total_calls}] {MODELS[model]['label']} "
                            f"{variant_label} · prompt {p_idx+1}/{len(prompts)} · run {run+1}"
                        )
                        if backend == "ollama":
                            res = bm.run_single(model, full_prompt, system=system, run_index=run)
                        else:
                            res = rag._run_cloud_model(model, system=system, prompt=full_prompt, run_index=run)
                        # Tag the result with the variant label so we can separate in stats
                        res.model = f"{model}{variant_label}" if variant_label else model
                        all_results.append(res)

        progress.empty(); status.empty()
        st.session_state.benchmark_results = all_results
        st.success(f"✅ {len(all_results)} benchmark runs complete")

    if st.session_state.benchmark_results:
        results = st.session_state.benchmark_results
        bm = Benchmarker()
        stats = bm.summary_stats(results)

        st.subheader("Summary Statistics (per model)")
        stats_rows = []
        for model_tag, s in stats.items():
            # model_tag may be "llama3.2:3b", "llama3.2:3b[Vector]", "llama3.2:3b[BM25]"
            variant = ""
            base_model = model_tag
            for tag in ("[Vector]", "[BM25]"):
                if model_tag.endswith(tag):
                    variant = f" {tag}"
                    base_model = model_tag[: -len(tag)]
                    break
            base_label = MODELS.get(base_model, {}).get("label", base_model)
            label = f"{base_label}{variant}"
            stats_rows.append({
                "Model":          label,
                "Runs":           s.get("runs", 0),
                "P50 lat (ms)":   s.get("p50_latency_ms", 0),
                "P95 lat (ms)":   s.get("p95_latency_ms", 0),
                "Min lat (ms)":   s.get("min_latency_ms", 0),
                "Avg TPS":        s.get("avg_tps", 0),
                "Avg TTFT (ms)":  s.get("avg_ttft_ms", 0),
                "Avg out tokens": s.get("avg_output_tokens", 0),
                "Cold starts":    s.get("cold_starts", 0),
                "Errors":         s.get("errors", 0),
            })
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

        # P50 vs P95 chart
        fig = go.Figure()
        for row in stats_rows:
            fig.add_trace(go.Bar(
                name=f"{row['Model']} P50",
                x=[row["Model"]],
                y=[row["P50 lat (ms)"]],
                marker_color="#3effa3",
            ))
            fig.add_trace(go.Bar(
                name=f"{row['Model']} P95",
                x=[row["Model"]],
                y=[row["P95 lat (ms)"]],
                marker_color="#ff6b6b",
            ))
        fig.update_layout(
            barmode="group",
            title="P50 vs P95 Latency (ms) — lower is better",
            template="plotly_dark",
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        # TPS chart
        tps_df = pd.DataFrame([
            {"Model": MODELS.get(m, {}).get("label", m), "Tokens/sec": s.get("avg_tps", 0)}
            for m, s in stats.items()
        ])
        fig2 = px.bar(tps_df, x="Model", y="Tokens/sec",
                      title="Average Tokens/Second — higher is better",
                      template="plotly_dark", color="Model")
        st.plotly_chart(fig2, use_container_width=True)

        # Raw results table
        with st.expander("Raw benchmark data"):
            raw_df = pd.DataFrame([
                {
                    "Model": MODELS.get(r.model, {}).get("label", r.model),
                    "Run": r.run_index,
                    "Total (ms)": round(r.total_duration_ms, 1),
                    "Load (ms)": round(r.load_duration_ms, 1),
                    "Prompt eval (ms)": round(r.prompt_eval_duration_ms, 1),
                    "Gen (ms)": round(r.eval_duration_ms, 1),
                    "TTFT (ms)": round(r.ttft_ms, 1),
                    "TPS": round(r.tokens_per_second, 1),
                    "Out tokens": r.output_tokens,
                    "Cold start": r.is_cold_start,
                    "Error": r.error or "—",
                }
                for r in results
            ])
            st.dataframe(raw_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# TAB 6 — TRADEOFF REPORT
# ════════════════════════════════════════════════════════════
with tab6:
    st.header("Quality vs Speed vs Cost — Tradeoff Report")
    st.caption(
        "Built from actual benchmark runs. Scores reflect YOUR hardware. "
        "Re-run benchmarks to refresh."
    )

    if not st.session_state.compare_results and not st.session_state.benchmark_results:
        st.info("Run a **Model Comparison** or **Benchmark Suite** first to populate this report.")
    else:
        # Use compare results if available, else synthesize from benchmarks
        if st.session_state.compare_results:
            results = st.session_state.compare_results
        else:
            # Aggregate benchmark results into pseudo-compare results
            from collections import defaultdict
            by_model = defaultdict(list)
            for r in st.session_state.benchmark_results:
                by_model[r.model].append(r)
            results = []
            for model, runs in by_model.items():
                good = [r for r in runs if not r.error]
                if not good:
                    continue
                avg_tps = sum(r.tokens_per_second for r in good) / len(good)
                avg_total = sum(r.total_duration_ms for r in good) / len(good)
                avg_ttft = sum(r.ttft_ms for r in good) / len(good)
                avg_gen = sum(r.eval_duration_ms for r in good) / len(good)
                avg_out = sum(r.output_tokens for r in good) // len(good)
                results.append({
                    "model": model,
                    "answer": good[0].response[:500] if good else "",
                    "retrieval_ms": 0,
                    "generation_ms": avg_gen,
                    "total_ms": avg_total,
                    "tokens_per_sec": avg_tps,
                    "ttft_ms": avg_ttft,
                    "output_tokens": avg_out,
                    "context_chunks": [],
                    "benchmark_result": good[0],
                })

        tradeoff_df = comparator.to_tradeoff_df(results)
        constraint_analysis = comparator.constraint_analysis(results)

        # ── Radar chart ──────────────────────────────────────
        st.subheader("Multi-dimensional Model Comparison")
        categories = ["Speed (/5)", "Latency (/5)", "Answer depth (/5)"]
        fig_radar = go.Figure()
        for _, row in tradeoff_df.iterrows():
            model_id = next((m for m, meta in MODELS.items() if meta["label"] == row["Model"]), None)
            color = MODELS.get(model_id, {}).get("color", "#ffffff") if model_id else "#ffffff"
            values = [row["Speed (/5)"], row["Latency (/5)"], row["Answer depth (/5)"]]
            values_closed = values + [values[0]]
            cats_closed = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=cats_closed,
                fill="toself",
                name=row["Model"],
                line_color=color,
                fillcolor=hex_to_rgba(color),
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0, 5])),
            template="plotly_dark",
            title="Model capabilities — your hardware",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Tradeoff scores table ─────────────────────────────
        st.subheader("Scoring Breakdown")
        st.dataframe(tradeoff_df, use_container_width=True, hide_index=True)

        # ── Constraint analysis ───────────────────────────────
        st.subheader("Real-world Constraint Analysis")
        st.caption("Privacy · Latency SLA · Cost — the constraints that matter for production use")

        c_cols = st.columns(len(constraint_analysis))
        for col, (model_label, analysis) in zip(c_cols, constraint_analysis.items()):
            with col:
                model_id = next((m for m, meta in MODELS.items() if meta["label"] == model_label), None)
                color = MODELS.get(model_id, {}).get("color", "#ffffff") if model_id else "#ffffff"
                st.markdown(f"<h3 style='color:{color}'>{model_label}</h3>", unsafe_allow_html=True)
                rows = [
                    ("🔒 Privacy", "✅ Pass" if analysis["privacy_ok"] else "❌ Fail"),
                    ("⚡ Interactive SLA (<3s)", "✅ Pass" if analysis["interactive_sla_ok"] else "❌ Fail"),
                    ("📦 Batch SLA (<10s)", "✅ Pass" if analysis["batch_sla_ok"] else "❌ Fail"),
                    ("💰 Per-query cost", analysis["cost_per_query"]),
                    ("🖥️ Est. RAM", f"~{analysis['est_ram_gb']} GB"),
                    ("📋 GDPR", "✅ Compliant"),
                    ("📡 Data egress", "None"),
                ]
                for k, v in rows:
                    st.markdown(f"**{k}:** {v}")

        # ── Written analysis ──────────────────────────────────
        st.divider()
        st.subheader("📋 Written Tradeoff Analysis")
        best_row = tradeoff_df.iloc[0] if not tradeoff_df.empty else None
        if best_row is not None:
            st.markdown(f"""
**Overall winner on this hardware:** **{best_row['Model']}** (score: {best_row['Overall score (/5)']}/5)

---

### Speed vs Quality Tradeoff
| Consideration | Insight |
|---|---|
| Fastest model | {tradeoff_df.sort_values('Speed (/5)', ascending=False).iloc[0]['Model']} |
| Lowest latency | {tradeoff_df.sort_values('Latency (/5)', ascending=False).iloc[0]['Model']} |
| Most thorough answers | {tradeoff_df.sort_values('Answer depth (/5)', ascending=False).iloc[0]['Model']} |
| Best overall score | {best_row['Model']} |

### When to pick each model
{chr(10).join(f"- **{row['Model']}:** {row['Best for']}" for _, row in tradeoff_df.iterrows())}

### Privacy & Cost Summary
Local (Ollama) models: 100% local, zero egress, $0/query, latency bounded by your hardware.
Free cloud models (OpenRouter): $0/query but rate-limited; retrieved chunks leave your device
and free tiers may log prompts. Their latency numbers include network and provider queue time,
so compare them with each other, not with local models.

### Latency vs Cloud API
| Scenario | Local SLM | GPT-4o API |
|---|---|---|
| Cost per 1M tokens | $0.00 | $12.50 avg |
| Network dependency | None | Required |
| Data residency | Local | US datacenters |
| Cold start | 1–5s | ~500ms |
| Warm inference | Model-dependent | ~1s typical |
| Context limit | 8k–128k | 128k |

**Verdict:** Local SLMs win on privacy and cost. Cloud wins on raw speed at warm state.
For document RAG with confidential content — local is the only viable choice.
            """)


# ════════════════════════════════════════════════════════════
# TAB 7 — RAGAS EVAL
# ════════════════════════════════════════════════════════════
with tab7:
    st.header("🧪 RAGAS Evaluation")
    st.caption(
        "Measure RAG output quality using RAGAS — faithfulness, answer relevancy, "
        "context precision, and more. Compare scores across multiple judge LLMs."
    )

    evaluator = RagasEvaluator()

    # ── Judge selector ────────────────────────────────────────
    st.subheader("🔵 Select Judges")
    st.caption(
        "Each selected judge runs independently and scores the same RAG output. "
        "More judges = richer comparison but longer runtime."
    )
    judge_cols = st.columns(len(JUDGE_OPTIONS))
    selected_judges: list[str] = []
    for col, (jkey, jmeta) in zip(judge_cols, JUDGE_OPTIONS.items()):
        with col:
            checked = st.checkbox(
                jmeta["label"],
                value=(jkey == "openrouter"),
                key=f"judge_chk_{jkey}",
            )
            if jmeta["warning"]:
                st.caption(f"⚠️ {jmeta['warning']}")
            if checked:
                selected_judges.append(jkey)

    if not selected_judges:
        st.warning("Select at least one judge to proceed.")

    st.divider()

    # ═══════════════════════════════════
    # SECTION A — Live single-query eval
    # ═══════════════════════════════════
    st.subheader("Section A — Live Query Evaluation (Reference-Free)")
    st.caption(
        "Type a question, run RAG, and score the output immediately. "
        "No golden dataset needed."
    )

    ea_col1, ea_col2 = st.columns([3, 1])
    with ea_col1:
        eval_question = st.text_input(
            "Question to evaluate",
            placeholder="e.g. What are the key findings in this document?",
            key="eval_q",
        )
    with ea_col2:
        eval_model = st.selectbox(
            "RAG model",
            options=st.session_state.selected_models,
            format_func=lambda m: MODELS[m]["label"],
            key="eval_model",
        )

    eval_topk = st.slider("Top-k chunks", 2, 10, 5, key="eval_topk")
    eval_gt = st.text_area(
        "Ground truth answer (optional — enables Context Recall, Answer Correctness)",
        placeholder="Leave empty for reference-free eval.",
        height=80,
        key="eval_gt",
    )

    if st.button("▶️ Run Live Eval", type="primary") and eval_question.strip():
        if rag.chunk_count == 0:
            st.warning("Index documents first (Upload & Index tab).")
        elif not selected_judges:
            st.warning("Select at least one judge above.")
        else:
            with st.spinner("Running RAG query..."):
                rag_result = rag.query(eval_question, eval_model, top_k=eval_topk)

            contexts = [c["text"] for c in rag_result["context_chunks"]]
            answer   = rag_result["answer"]
            ground_t = eval_gt.strip() if eval_gt.strip() else None

            # Display the generated answer
            st.markdown("**Generated answer:**")
            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True
            )

            # Run each judge
            st.subheader("RAGAS Scores")
            score_cols = st.columns(len(selected_judges))
            all_judge_scores: dict[str, dict] = {}

            for col, judge in zip(score_cols, selected_judges):
                jlabel = JUDGE_OPTIONS[judge]["label"]
                with col:
                    st.markdown(f"**{jlabel}**")
                    with st.spinner(f"Judging with {jlabel}..."):
                        try:
                            scores = evaluator.evaluate_single(
                                judge=judge,
                                question=eval_question,
                                answer=answer,
                                contexts=contexts,
                                ground_truth=ground_t,
                            )
                            all_judge_scores[judge] = scores
                        except Exception as e:
                            st.error(f"Error: {e}")
                            scores = {}

                    error_msg = scores.pop("error", None)
                    eval_time = scores.pop("eval_time_s", None)

                    for metric, val in scores.items():
                        if isinstance(val, float) and val != val:  # NaN
                            st.metric(metric, "❌ NaN", help="Judge failed to score — check warnings above")
                        else:
                            pct = round(val * 100) if isinstance(val, float) else val
                            color = "normal" if pct >= 70 else ("off" if pct >= 40 else "inverse")
                            st.metric(metric, f"{pct}%")

                    if eval_time:
                        st.caption(f"⏱️ {eval_time}s")
                    if error_msg:
                        st.warning(f"Judge warning: {error_msg}")

            # Judge agreement (if > 1 judge selected)
            if len(all_judge_scores) > 1:
                st.divider()
                st.subheader("Judge Agreement")
                common_metrics = set.intersection(
                    *[set(s.keys()) for s in all_judge_scores.values()]
                ) - {"eval_time_s", "error"}
                agree_rows = []
                for metric in sorted(common_metrics):
                    vals = {j: all_judge_scores[j].get(metric, float("nan"))
                             for j in all_judge_scores}
                    valid = [v for v in vals.values() if v == v]
                    spread = round(max(valid) - min(valid), 3) if len(valid) > 1 else None
                    row = {"Metric": metric, "Spread (max-min)": spread}
                    row.update({JUDGE_OPTIONS[j]["label"]: f"{round(v*100)}%" if v==v else "NaN"
                                 for j, v in vals.items()})
                    agree_rows.append(row)
                st.dataframe(pd.DataFrame(agree_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ═══════════════════════════════════
    # SECTION B — Batch golden dataset
    # ═══════════════════════════════════
    st.subheader("Section B — Batch Golden Dataset Evaluation")
    st.caption(
        "Upload a CSV with `question` and `ground_truth` columns. "
        "The app will query RAG for each row and evaluate with selected judges."
    )

    with st.expander("📋 CSV format"):
        st.code(
            "question,ground_truth\n"
            '"What is the main topic?","The document covers renewable energy."\n'
            '"Who authored the report?","The report was authored by Dr. Smith."',
            language="csv"
        )

    uploaded_csv = st.file_uploader(
        "Upload evaluation CSV", type=["csv"], key="eval_csv"
    )
    batch_model = st.selectbox(
        "RAG model for batch eval",
        options=st.session_state.selected_models,
        format_func=lambda m: MODELS[m]["label"],
        key="batch_eval_model",
    )
    batch_topk = st.slider("Top-k (batch)", 2, 10, 5, key="batch_topk")

    if uploaded_csv is not None:
        import io
        batch_df = pd.read_csv(io.StringIO(uploaded_csv.getvalue().decode("utf-8")))
        has_gt   = "ground_truth" in batch_df.columns

        st.info(
            f"📊 Loaded **{len(batch_df)} rows**"
            + (" · includes `ground_truth` → full metric set enabled" if has_gt
               else " · no `ground_truth` column → reference-free metrics only")
        )
        st.dataframe(batch_df.head(5), use_container_width=True, hide_index=True)

        if st.button("▶️ Run Batch Eval", type="primary"):
            if not selected_judges:
                st.warning("Select at least one judge above.")
            else:
                total = len(batch_df)
                prog  = st.progress(0)
                stat  = st.empty()

                rows = []
                for i, row in batch_df.iterrows():
                    stat.text(f"[{i+1}/{total}] RAG query for: {str(row['question'])[:60]}...")
                    rag_r = rag.query(str(row["question"]), batch_model, top_k=batch_topk)
                    rows.append({
                        "question":     str(row["question"]),
                        "answer":       rag_r["answer"],
                        "contexts":     [c["text"] for c in rag_r["context_chunks"]],
                        "ground_truth": str(row.get("ground_truth", "")) if has_gt else None,
                    })
                    prog.progress((i + 1) / total)

                prog.empty(); stat.empty()

                with st.spinner(f"Scoring {total} rows across {len(selected_judges)} judge(s)..."):
                    result_df = evaluator.evaluate_dataset(
                        judges=selected_judges,
                        rows=rows,
                    )

                st.session_state.eval_results = result_df
                st.success(f"✅ Batch eval complete — {len(result_df)} rows scored")

    if st.session_state.eval_results is not None:
        result_df = st.session_state.eval_results
        st.subheader("Batch Results (per row)")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        # Summary: mean per metric column
        metric_cols = [c for c in result_df.columns if c != "question"]
        summary = result_df[metric_cols].mean(numeric_only=True).reset_index()
        summary.columns = ["Metric / Judge", "Mean Score"]
        summary["Mean Score"] = summary["Mean Score"].apply(
            lambda v: f"{round(v*100)}%" if v == v else "NaN"
        )

        st.subheader("Summary (Mean across all rows)")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Radar chart
        radar_data = result_df[metric_cols].mean(numeric_only=True).dropna()
        if len(radar_data) >= 3:
            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_data.values.tolist() + [radar_data.values[0]],
                theta=radar_data.index.tolist() + [radar_data.index[0]],
                fill="toself",
                name="Mean scores",
                line_color="#4ECDC4",
                fillcolor="rgba(78,205,196,0.2)",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="RAGAS Score Radar (mean across rows)",
                template="plotly_dark",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        if st.button("🗑️ Clear eval results"):
            st.session_state.eval_results = None
            st.rerun()

