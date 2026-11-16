# ============================================================
# config.py — Central configuration for Local SLM RAG App
# ============================================================

import os
from dotenv import load_dotenv

# Load keys from .env file (sits next to this file)
# Environment variables already set take precedence (override=False)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

OLLAMA_BASE_URL = "http://localhost:11434"

# ── OpenRouter (default backend — free cloud models) ────────
# One key gives access to hundreds of models, including free ones
# (IDs ending in ":free"). Get a key at https://openrouter.ai/keys
#
# ⚠️  Free models change often. To swap them, set OPENROUTER_MODELS in .env
#     as a comma-separated list, e.g.
#     OPENROUTER_MODELS=google/gemma-4-26b-a4b-it:free,liquid/lfm-3-1.7b:free
#     The Setup tab lists the free models available right now.
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_OPENROUTER_PALETTE = ["#6C8EF5", "#F57CB0", "#5CD6A9", "#F5B95C", "#B48EFF", "#5CC8F5"]

# Known metadata for the default free models (checked 2026-11-16).
# Gemma 4 26B is popular and often rate-limited upstream — expect retries.
_OPENROUTER_KNOWN = {
    "liquid/lfm-3-1.7b:free": {
        "label": "LFM-3 1.7B", "family": "Liquid AI", "params_b": 1.7, "context_len": 262_144,
    },
    "google/gemma-4-26b-a4b-it:free": {
        "label": "Gemma 4 26B-A4B", "family": "Google", "params_b": 26.0, "context_len": 262_144,
    },
    "nvidia/nemotron-3.5-lightning:free": {
        "label": "Nemotron 3.5 Lightning", "family": "NVIDIA", "params_b": 0.0, "context_len": 1_000_000,
    },
    "baidu/ernie-5.5-flash-preview:free": {
        "label": "ERNIE 5.5 Flash", "family": "Baidu", "params_b": 0.0, "context_len": 1_024_000,
    },
    "cohere/north-mini-code:free": {
        "label": "North Mini Code", "family": "Cohere", "params_b": 0.0, "context_len": 256_000,
    },
    "recursive/flint-v1.0:free": {
        "label": "Flint v1.0", "family": "Recursive", "params_b": 0.0, "context_len": 262_144,
    },
}

_DEFAULT_OPENROUTER_MODELS = list(_OPENROUTER_KNOWN.keys())

# Judge model for RAGAS evals (should be a larger model — judges need to
# return well-formed JSON). Override with OPENROUTER_JUDGE_MODEL in .env.
OPENROUTER_JUDGE_MODEL = os.getenv(
    "OPENROUTER_JUDGE_MODEL", "nvidia/nemotron-3.5-lightning:free"
)


def _openrouter_models() -> dict:
    """Build MODELS entries for OpenRouter from .env (or the defaults)."""
    raw = os.getenv("OPENROUTER_MODELS", "").strip()
    ids = [m.strip() for m in raw.split(",") if m.strip()] if raw else _DEFAULT_OPENROUTER_MODELS
    out = {}
    for i, model_id in enumerate(ids):
        known = _OPENROUTER_KNOWN.get(model_id, {})
        out[model_id] = {
            "label": "🌐 " + known.get("label", model_id.split("/")[-1].replace(":free", "")),
            "color": _OPENROUTER_PALETTE[i % len(_OPENROUTER_PALETTE)],
            "params_b": known.get("params_b", 0.0),
            "context_len": known.get("context_len", 0),
            "family": known.get("family", model_id.split("/")[0]),
            "quantization": "Cloud (managed)",
            "disk_size_gb": 0.0,
            "pull_cmd": "# No pull needed — OpenRouter API",
            "backend": "openrouter",
            "data_local": False,
        }
    return out


# ── Models under comparison ─────────────────────────────────
# OpenRouter free models first (default), then optional local Ollama models,
# then other optional cloud APIs.
MODELS = {
    **_openrouter_models(),

    # ── Local models (optional — need Ollama + `ollama pull`) ──
    # ── Local models (via Ollama) ────────────────────────────
    "llama3.2:3b": {
        "label": "Llama 3.2 3B",
        "color": "#FF6B6B",
        "params_b": 3.2,
        "context_len": 128_000,
        "family": "Meta",
        "quantization": "Q4_K_M (default)",
        "disk_size_gb": 2.0,
        "pull_cmd": "ollama pull llama3.2:3b",
        "backend": "ollama",
        "data_local": True,
    },
    "gemma2:2b": {
        "label": "Gemma 2 2B",
        "color": "#4ECDC4",
        "params_b": 2.0,
        "context_len": 8_192,
        "family": "Google",
        "quantization": "Q4_K_M (default)",
        "disk_size_gb": 1.6,
        "pull_cmd": "ollama pull gemma2:2b",
        "backend": "ollama",
        "data_local": True,
    },
    "qwen3:4b": {
        "label": "Qwen3 4B",
        "color": "#FFD93D",
        "params_b": 4.0,
        "context_len": 32_768,
        "family": "Alibaba",
        "quantization": "Q4_K_M (default)",
        "disk_size_gb": 2.6,
        "pull_cmd": "ollama pull qwen3:4b",
        "backend": "ollama",
        "data_local": True,
    },
    "phi4-mini:latest": {
        "label": "Phi-4 Mini",
        "color": "#A8E6CF",
        "params_b": 3.8,
        "context_len": 16_384,
        "family": "Microsoft",
        "quantization": "Q4_K_M (default)",
        "disk_size_gb": 2.5,
        "pull_cmd": "ollama pull phi4-mini",
        "backend": "ollama",
        "data_local": True,
    },

    # ── Cloud API models ─────────────────────────────────────
    # ⚠️  Data leaves device. Not for confidential documents.
    "moonshotai/kimi-k2-instruct": {
        "label": "☁️ Kimi K2 Instruct",
        "color": "#B48EFF",
        "params_b": 32.0,          # active params (MoE, 1T total)
        "context_len": 256_000,
        "family": "Moonshot AI",
        "quantization": "Cloud (managed)",
        "disk_size_gb": 0.0,       # no local disk usage
        "pull_cmd": "# No pull needed — cloud API",
        "backend": "nvidia_nim",
        "data_local": False,
    },
    "google/gemini-2.5-flash": {
        "label": "☁️ Gemini 2.5 Flash",
        "color": "#FF9F45",
        "params_b": 0.0,           # not disclosed
        "context_len": 1_000_000,
        "family": "Google DeepMind",
        "quantization": "Cloud (managed)",
        "disk_size_gb": 0.0,
        "pull_cmd": "# No pull needed — cloud API",
        "backend": "gemini",
        "data_local": False,
    },
}

# ── RAG settings ─────────────────────────────────────────────
# Default embedding model: runs locally on CPU via sentence-transformers
# (~90 MB download on first use, then offline). No Ollama needed.
# "nomic-embed-text" (Ollama) is still available as an option.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384          # all-MiniLM-L6-v2 output dim

CHUNK_SIZE = 500             # characters per chunk
CHUNK_OVERLAP = 60           # overlap to preserve context at boundaries
TOP_K_RETRIEVAL = 5          # chunks retrieved per query

# ── Benchmark settings ───────────────────────────────────────
BENCHMARK_PROMPTS = [
    "Summarize the key points from the document.",
    "What are the main conclusions or recommendations?",
    "List any numerical data, statistics, or metrics mentioned.",
    "What problem is this document trying to solve?",
    "Who is the intended audience of this document?",
]

# ── Cloud API configuration ──────────────────────────────────
# Set via environment variables — never hardcode keys here!
#
# For NVIDIA NIM (Kimi K2):  set NVIDIA_API_KEY env var
# For Gemini:                set GEMINI_API_KEY env var
#
# Get NVIDIA NIM key (free): https://build.nvidia.com
# Get Gemini key:            https://aistudio.google.com

NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = "gemini-2.5-flash"   # Gemini 2.5 Flash

# ── Cost reference (per 1M tokens) ───────────────────────────
API_COST_REFERENCE = {
    "GPT-4o (OpenAI)":           {"input": 2.50,  "output": 10.00},
    "Claude 3.5 Sonnet":         {"input": 3.00,  "output": 15.00},
    "Gemini 1.5 Pro":            {"input": 1.25,  "output": 5.00},
    "☁️ Kimi K2 Instruct":      {"input": 0.60,  "output": 2.50},
    "☁️ Gemini 3":              {"input": 0.00,  "output": 0.00},  # TBD
    "Local (any model)":         {"input": 0.00,  "output": 0.00},
    "OpenRouter free models":    {"input": 0.00,  "output": 0.00},
}

# ── Rate limiting / retries for cloud APIs ───────────────────
# Free tiers return HTTP 429 when you go too fast. We retry with
# exponential backoff (honouring Retry-After when the API sends it).
API_MAX_RETRIES = 3
API_BACKOFF_BASE_S = 4.0     # 4s, 8s, 16s ...
API_BACKOFF_MAX_S = 30.0
API_TIMEOUT_S = 120.0        # per attempt; free-tier queues can be slow
# Output budget per answer. Reasoning ("thinking") models spend part of this
# on hidden reasoning before the visible answer (Nemotron 3.5 Lightning used
# up to ~1,000 tokens thinking on short RAG prompts), so keep it generous.
# If an answer still comes back empty, the client retries once with double
# this budget and low reasoning effort.
API_MAX_OUTPUT_TOKENS = 4096

# ── RAGAS judge pacing ───────────────────────────────────────
# RAGAS defaults to 16 parallel judge calls — far above free-tier limits.
RAGAS_MAX_WORKERS = 2
RAGAS_TIMEOUT_S = 180
RAGAS_MAX_RETRIES = 6
RAGAS_MAX_WAIT_S = 60

# ── Persistence ──────────────────────────────────────────────
# Keep the document index and Learn progress on disk so they survive
# restarts. Stored under data/ (gitignored). Set PERSIST_INDEX=false in
# .env for a throwaway in-memory index.
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PERSIST_INDEX = os.getenv("PERSIST_INDEX", "true").strip().lower() not in ("0", "false", "no")
INDEX_DIR = os.path.join(_DATA_DIR, "chroma_db")
LEARN_PROGRESS_FILE = os.path.join(_DATA_DIR, "learn_progress.json")

# ── Indexing ─────────────────────────────────────────────────
# ChromaDB rejects inserts above ~5,461 items, so large documents are
# embedded and stored in batches of this size.
INDEX_BATCH_SIZE = 512

# ── Privacy settings ─────────────────────────────────────────
# Only true for local models (Ollama). Cloud models (OpenRouter, NVIDIA,
# Gemini) send your retrieved document chunks to a third party, and free
# tiers may log prompts. Use non-sensitive documents with cloud models.
PRIVACY_CONFIG = {
    "network_calls_blocked": False,
    "data_leaves_device": True,      # with the default (OpenRouter) models
    "logs_sent_to_vendor": True,
    "gdpr_compliant": False,
    "telemetry": False,              # chromadb telemetry disabled
}
