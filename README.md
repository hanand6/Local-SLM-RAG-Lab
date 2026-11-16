# 🧠 SLM RAG Lab

A learning lab for Document Q&A (RAG), model comparison and evaluation.
It runs on **free OpenRouter models** by default, so you need no GPU, no model downloads and no Ollama.
Local Ollama models are still supported as an option.

> ⚠️ **Privacy:** with OpenRouter (the default), the document chunks retrieved for each question are sent to a third-party API, and free tiers may log prompts. Use non-sensitive documents. Embeddings and the vector store always run locally. For fully offline use, switch to local Ollama models (see below).

---

## What This Project Does

| Feature | Detail |
|---|---|
| Document Q&A | Upload PDFs/DOCX/TXT/MD → ask questions → get grounded answers with sources |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers, local on CPU (~90 MB, downloaded once) |
| Vector store | ChromaDB (in-process, no Docker, no server) |
| Generation | Free OpenRouter models (default) · optional Ollama, NVIDIA NIM, Gemini |
| Model comparison | Same question → several models → side-by-side answers + metrics |
| Benchmarking | P50/P95 latency, tokens/sec, TTFT |
| Retrieval | Vector search, optional BM25 reranking side by side |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision, …) with an LLM judge |

---

## Setup

### 1. Install Python dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get a free OpenRouter key

Create a key at <https://openrouter.ai/keys>. Then copy `.env.example` to `.env` and paste the key in:

```
OPENROUTER_API_KEY=sk-or-...
```

### 3. Launch the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. The first time you index a document it downloads the embedding model (~90 MB).

---

## Choosing models

The default free models live in `config.py`. Free models are renamed or removed often, so:

- The **Setup** tab lists the free models available right now and warns you if a configured one has disappeared.
- To swap models without touching code, set this in `.env` and restart:

  ```
  OPENROUTER_MODELS=google/gemma-4-26b-a4b-it:free,liquid/lfm-3-1.7b:free
  ```

- The RAGAS judge model is set separately with `OPENROUTER_JUDGE_MODEL`. Pick a larger model, because judges must return well-formed JSON.

### Free-tier limits

Free models allow roughly **20 requests/minute** plus a **daily cap**. The daily cap is higher once your account has a small credit balance. Check OpenRouter's docs for current numbers. The app:

- retries rate-limited (HTTP 429) and server-error calls with exponential backoff, using `Retry-After` when the API sends it
- shows a clear message when a model is missing (404), the key is wrong (401) or you're still rate-limited after retries

Keep benchmark and eval runs small: the Benchmark tab shows how many calls a run will make before you start it.

---

## Optional: local models with Ollama

For fully offline, private inference:

```bash
ollama serve
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull qwen3:4b
ollama pull phi4-mini
ollama pull nomic-embed-text   # optional alternative embedding model
```

Local models appear in the sidebar and are ticked automatically once pulled. When only local models are selected, no document data leaves your device.

---

## Project Structure

```
local-slm-rag/
├── app.py              ← Streamlit UI (tabs: Setup, Upload, Ask, Compare, Benchmark, Tradeoff, Eval)
├── config.py           ← Models (OpenRouter / Ollama / other cloud), RAG settings, retry settings
├── api_client.py       ← OpenAI-compatible client (OpenRouter, NVIDIA) with retries + Gemini client
├── ollama_client.py    ← Ollama REST API wrapper (optional local models)
├── embedder.py         ← Embeddings: sentence-transformers (default) or Ollama
├── rag_engine.py       ← Loading, chunking, embedding, retrieval, generation routing
├── reranker.py         ← BM25 reranking
├── benchmarker.py      ← Latency / TPS / TTFT metrics
├── comparator.py       ← Multi-model comparison + tradeoff scoring
├── ragas_evaluator.py  ← RAGAS evaluation with selectable judge LLMs
├── requirements.txt
└── .env.example        ← Template for API keys and model overrides
```

---

## Architecture

```
User uploads document
       │
       ▼
  load_document()          ← pypdf / python-docx / plain text
       │
       ▼
  TextChunker.split()      ← 500-char chunks, 60-char overlap
       │
       ▼
  EmbeddingClient          ← all-MiniLM-L6-v2 (local CPU)
       │
       ▼
  ChromaDB.add()           ← in-process vector store
       │
  ─────────────── query time ───────────────
       │
  embed question → ChromaDB.query() → [optional BM25 rerank]
       │
       ▼
  build_prompt()           ← system rules + retrieved context + question
       │
       ▼
  generate                 ← OpenRouter (default) · Ollama · NVIDIA · Gemini
       │
       ▼
  BenchmarkResult          ← answer, timing, tokens, retries
```

---

## Reading the benchmark numbers

| Metric | Local (Ollama) | Cloud (OpenRouter) |
|---|---|---|
| TTFT | Prefill + model load time | Network + provider queue + prefill |
| Tokens/sec | Your hardware's speed | Provider speed, varies with load |
| Token counts | Exact (Ollama) | Exact when the API reports usage, else estimated |
| Cold/warm start | Meaningful | Not applicable |
| RAM/Disk | Meaningful | Not applicable |

Compare cloud models with each other, not with local ones. Their latency mostly reflects network and queueing, not model speed.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `OPENROUTER_API_KEY not set` | Add it to `.env` and restart the app |
| `Model ... was not found (404)` | Free model was removed; check the Setup tab and update `OPENROUTER_MODELS` |
| `Rate limited (429)` | Wait a minute, run fewer calls, or switch model |
| First indexing is slow | One-time download of the embedding model (~90 MB) |
| Local model not available | `ollama serve` and `ollama pull <model>` |

---

## License

MIT — use freely for personal and commercial projects.
