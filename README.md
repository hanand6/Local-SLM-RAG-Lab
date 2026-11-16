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
| 🎓 Learn | Guided chapters for AI PM interview topics: concept → hands-on lab on your document → interview card |

---

## 🎓 Learn chapters

Each chapter has three parts:
- **Concept:** a short read.
- **Lab:** runs on your own indexed document and shows how many model calls it will make.
- **Interview:** practise a committed answer, then reveal a model answer with pitfalls.

| # | Chapter | Lab |
|---|---|---|
| 1 | Prompting vs RAG vs fine-tuning | Same question three ways (prompt only / whole document / RAG), plus a decision drill |
| 2 | Hallucination and grounding | Strict vs loose prompt, abstention threshold, citation and support checks |
| 3 | Evals | Golden set + LLM-as-judge vs your own grades; cost-based threshold simulator |
| 4 | Latency and cost | Live streaming, exact + semantic cache, small/large model routing, build-vs-buy break-even |
| 5 | Agents and safeguards | A tool-using agent with an approval policy, recipient allowlist and step budget; error-compounding chart; "when not to" drill |
| 6 | Classic ML literacy | Why accuracy lies on imbalanced data; rules vs TF-IDF classifier vs LLM on ticket routing; "which approach" drill |
| 7 | AI system design | This app's architecture as a worked example; a design canvas that draws your diagram and gets an AI critique |
| 8 | Launches and responsible AI | Staged-rollout simulator with guardrails and hidden failure scenarios; name-swap bias probe; EU AI Act risk-tier drill |

Each chapter's interview card covers the matching question type from AI PM interview prep. The order builds up: ML literacy comes before system design, and launches come last.

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

## Using the 🎓 Learn tab

### 1. Check the app is ready

Open the **⚙️ Setup** tab. You should see **"OPENROUTER_API_KEY is set"**, and the sidebar should show **OpenRouter: 🟢 Key set**. If not, check `.env` and restart the app. Ollama is optional, so a "not running" status is fine.

### 2. Index a document

Most labs run on your own document, so index one first:

1. Open **📄 Upload & Index**.
2. Drop in a file, or use the included sample: [`samples/acme_annual_report_2025.md`](samples/acme_annual_report_2025.md). It's a fictional annual report with numbers, risks and recommendations, which suits every lab.
3. Click **📥 Index …** and wait for **"You're ready for the next step"**. The first run takes about 20s while the embedding model loads.

Use non-sensitive documents: retrieved chunks are sent to the free cloud models.

### 3. Work through a chapter

1. Open **🎓 Learn** and pick a chapter on the left. They build on each other, so start with chapter 1.
2. **📖 Concept:** read it first (about 5 minutes).
3. **🧪 Lab:** each lab says how many model calls it makes before you run it.
4. **🎤 Interview:** write your answer in the box (or say it out loud) *before* clicking **Reveal**.
5. Tick **Mark this chapter complete**.

### What each lab needs

| Chapter | Needs an indexed document? | Model calls per run |
|---|---|---|
| 1. Prompting vs RAG vs fine-tuning | Yes | 3 |
| 2. Hallucination and grounding | Yes | 1–2 |
| 3. Evals | Lab A: yes · Lab B: no | A: 2 per question (max 10) · B: 0 |
| 4. Latency and cost | Labs A–C: yes · Lab D: no | A: 1 · B: 0–1 · C: 2 · D: 0 |
| 5. Agents and safeguards | Lab A: yes · Labs B–C: no | A: up to 7 · B, C: 0 |
| 6. Classic ML literacy | No | B: 1 (rules and ML run locally) · A, C: 0 |
| 7. AI system design | No | B: 0–1 (the critique is optional) |
| 8. Launches and responsible AI | No | B: 2–8 · A, C: 0 |

### Tips

- **Models:** labs offer only the models ticked in the sidebar. Flint v1.0 and Nemotron 3.5 Lightning are fast and reliable defaults. Gemma 4 26B is often rate-limited.
- **Rate limits:** free models allow roughly 20 calls a minute. The call counter on the left of the Learn tab tracks your session. If you hit a limit, wait a minute; calls are retried automatically.
- **Good questions for the sample report:** *"What does the board recommend for 2026?"* (answerable), *"What was revenue in 1995?"* (unanswerable, good for chapter 2: it still scores about 0.41 similarity, so raise the abstain threshold to 0.45 to see the gate refuse it), *"How much did logistics costs grow?"* (a number, good for chapter 3's golden set).
- **Progress and lab results** are kept for the browser session only. A page refresh or app restart clears them, along with the index, so re-index after restarting.

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

Keep benchmark and eval runs small: the Benchmark tab shows how many calls a run will make before you start it. RAGAS judge calls run 2 at a time (not RAGAS's default 16) to stay under the limits.

### Reasoning models

Some free models "think" before answering (for example LFM-3, Nemotron, North Mini Code and Flint). That thinking counts as output tokens, so the app gives each answer a 4,096-token budget (`API_MAX_OUTPUT_TOKENS`). If a model still returns an empty answer, the app retries once with double the budget and low reasoning effort. Results report reasoning tokens separately, plus the time until the first visible word of the answer.

---

## Document size

Rate limits don't depend on document size. Indexing is fully local (embeddings on your CPU), and each question sends only the top-k chunks (~2.5 KB) to the model. What does grow with size is **indexing time**:

| Text in document | Chunks (approx.) | Indexing time (measured, CPU) |
|---|---|---|
| 1 MB | ~1,300 | ~25 s |
| 5 MB | ~6,500 | ~2 min |
| 30 MB | ~40,000 | ~15–20 min |

- The size that matters is the **text**, not the file. A 30 MB PDF full of images may hold only 1–2 MB of text.
- Indexing runs in batches with a progress bar. The upload limit is Streamlit's default 200 MB.
- The index is in memory, so it is cleared when the app restarts.
- RAG answers questions from a few chunks. It can't "summarize the whole document" for large files.

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
├── learn/              ← 🎓 Learn tab: ui.py + one module per chapter
├── samples/            ← Fictional sample document for the labs
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
