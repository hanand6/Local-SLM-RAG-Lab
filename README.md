# 🧠 Local SLM RAG Lab

A fully offline Document Q&A system with inference benchmarking and model comparison.  
**Zero external API calls. $0 inference cost. 100% private.**

---

## What This Project Does

| Feature | Detail |
|---|---|
| Document Q&A | Upload PDFs/DOCX/TXT → ask questions → get grounded answers |
| Local embeddings | `nomic-embed-text` via Ollama — no OpenAI embeddings API |
| Vector store | ChromaDB (in-process, no Docker, no server) |
| Model comparison | Same question → 4 models → side-by-side answers + metrics |
| Inference benchmarking | P50/P95 latency, tokens/sec, TTFT, cold vs warm start |
| Tradeoff report | Quality × Speed × Cost scoring with radar chart |
| Privacy proof | All traffic on `localhost:11434`, telemetry disabled |

---

## Models Compared

| Model | Family | Params | Context | Disk | Est. RAM |
|---|---|---|---|---|---|
| `llama3.2:3b` | Meta | 3.2B | 128k | ~2.0 GB | ~1.9 GB |
| `gemma2:2b` | Google | 2.0B | 8k | ~1.6 GB | ~1.2 GB |
| `qwen3:4b` | Alibaba | 4.0B | 32k | ~2.6 GB | ~2.4 GB |
| `phi4-mini` | Microsoft | 3.8B | 16k | ~2.5 GB | ~2.3 GB |

> RAM estimates based on Q4_K_M quantization (~0.6 GB per billion parameters).  
> All models run entirely on-device. No cloud dependency after initial pull.

---

## Setup

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: download from https://ollama.ai
```

### 2. Pull models

```bash
ollama pull llama3.2:3b        # ~2.0 GB
ollama pull gemma2:2b          # ~1.6 GB
ollama pull qwen3:4b           # ~2.6 GB
ollama pull phi4-mini          # ~2.5 GB
ollama pull nomic-embed-text   # ~274 MB — required for embeddings
```

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Start Ollama (if not already running)

```bash
ollama serve
# Runs on http://localhost:11434
```

### 5. Launch the app

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## Project Structure

```
local-slm-rag/
├── app.py              ← Streamlit UI (6 tabs)
├── config.py           ← Model definitions, RAG settings, cost reference
├── ollama_client.py    ← Ollama REST API wrapper (privacy-enforced)
├── rag_engine.py       ← Document ingestion, chunking, embedding, retrieval
├── benchmarker.py      ← Inference metrics: TPS, TTFT, P50/P95 latency
├── comparator.py       ← Multi-model comparison + tradeoff scoring
├── requirements.txt
├── README.md
├── uploads/            ← Temp dir for uploaded files
├── data/               ← (optional) pre-loaded documents
└── reports/            ← (optional) saved benchmark exports
```

---

## Architecture

```
User uploads PDF
       │
       ▼
  load_document()         ← pypdf / python-docx / plain text
       │
       ▼
  TextChunker.split()     ← 500-char chunks, 60-char overlap
       │
       ▼
  OllamaClient.embed()    ← nomic-embed-text @ localhost:11434
       │
       ▼
  ChromaDB.add()          ← in-process cosine similarity store
       │
  ─────────────── query time ───────────────
       │
  User asks question
       │
       ▼
  OllamaClient.embed()    ← embed the question
       │
       ▼
  ChromaDB.query()        ← top-5 most similar chunks
       │
       ▼
  build_prompt()          ← system + context + question
       │
       ▼
  OllamaClient.generate() ← LLM inference @ localhost:11434
       │
       ▼
  BenchmarkResult         ← timing, TPS, TTFT captured
       │
       ▼
  Streamlit UI            ← answer + metrics displayed
```

---

## Benchmarking Methodology

### Metrics captured from Ollama's native timing API

| Metric | Source | What it measures |
|---|---|---|
| `total_duration` | Ollama native | Full wall time including queue |
| `load_duration` | Ollama native | Model load time (>200ms = cold start) |
| `prompt_eval_duration` | Ollama native | Tokenization + prefill (KV cache build) |
| `eval_duration` | Ollama native | Token generation time |
| `eval_count` | Ollama native | Number of output tokens |
| **TTFT** | Derived | `prompt_eval_duration + load_duration (if cold)` |
| **TPS** | Derived | `eval_count / (eval_duration / 1000)` |
| **P50/P95** | Computed | Across N runs per model |

### Cold vs Warm starts

- **Cold start:** Model not in RAM — Ollama loads from disk. `load_duration > 200ms`.
- **Warm start:** Model already loaded — `load_duration ≈ 0`.
- First run per session is always cold. Subsequent runs are warm.

---

## Privacy Architecture

### What happens to your data

```
Document uploaded → processed in Python memory → chunked → embedded locally
Query asked       → embedded locally → retrieved locally → answered locally
                  
All traffic:  localhost:11434 (Ollama)
External calls: ZERO
Data at rest: ChromaDB in-memory (cleared on restart)
Telemetry: disabled (chromadb anonymized_telemetry=False)
```

### Compliance posture

| Regulation | Status | Reason |
|---|---|---|
| GDPR | ✅ Compliant | No personal data leaves device |
| HIPAA | ✅ Compatible | No PHI transmitted externally |
| SOC 2 | ✅ Compatible | No third-party data access |
| CCPA | ✅ Compliant | No data sold or shared |

---

## Cost Analysis

### Local vs Cloud API (1000 queries/day, avg 2000 tokens each)

| Provider | Input cost/day | Output cost/day | Monthly total |
|---|---|---|---|
| GPT-4o | ~$3.50 | ~$7.00 | **~$315** |
| Claude 3.5 Sonnet | ~$4.20 | ~$10.50 | **~$441** |
| Gemini 1.5 Pro | ~$1.75 | ~$3.50 | **~$158** |
| **Local SLM** | **$0** | **$0** | **$0** |

> Local cost = electricity only (≈ $0.02/day on a laptop).

---

## Latency Profile

### Typical latency on common hardware (warm start, 500-token output)

| Hardware | Llama 3.2 3B | Gemma 2 2B | Qwen3 4B | Phi-4 Mini |
|---|---|---|---|---|
| Apple M2 Pro | ~1.2s | ~0.8s | ~1.8s | ~1.5s |
| Apple M1 | ~2.0s | ~1.3s | ~3.0s | ~2.5s |
| RTX 3080 (GPU) | ~0.8s | ~0.5s | ~1.2s | ~1.0s |
| Intel i7 (CPU only) | ~8s | ~5s | ~12s | ~10s |

> Values are approximate. Run the benchmark suite to get your actual numbers.

### Latency vs Cloud

| Metric | Local SLM | GPT-4o API |
|---|---|---|
| Cold start | 2–10s | ~200ms |
| Warm TTFT | 200–800ms | ~400ms |
| Network dep. | None | Required (50–200ms RTT) |
| Tail latency | Predictable | Variable (shared infra) |

---

## Extending the Project

### Add a new model

```python
# In config.py, add to MODELS dict:
"mistral:7b": {
    "label": "Mistral 7B",
    "color": "#C678DD",
    "params_b": 7.0,
    "context_len": 32_768,
    "family": "Mistral AI",
    "quantization": "Q4_K_M",
    "disk_size_gb": 4.1,
    "pull_cmd": "ollama pull mistral:7b",
},
```

### Persist the vector store across sessions

```python
# In rag_engine.py, replace in-memory client with persistent:
self._chroma = chromadb.PersistentClient(
    path="./data/chroma_db",
    settings=Settings(anonymized_telemetry=False)
)
```

### Export benchmark results to CSV

```python
import pandas as pd
df = pd.DataFrame([r.to_dict() for r in benchmark_results])
df.to_csv("reports/benchmark_results.csv", index=False)
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ConnectionError: localhost:11434` | Run `ollama serve` in a terminal |
| `Model not found` | Run `ollama pull <model>` |
| `chromadb` import error | `pip install chromadb` |
| Embedding fails | Run `ollama pull nomic-embed-text` |
| Slow on CPU | Expected — try a 2B model first; GPU acceleration recommended |
| Out of memory | Use smaller models (gemma2:2b); close other apps |

---

## License

MIT — use freely for personal and commercial projects.
