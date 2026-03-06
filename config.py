# ============================================================
# config.py — Central configuration for Local SLM RAG App
# ============================================================

import os
from dotenv import load_dotenv

# Load keys from .env file (sits next to this file)
# Environment variables already set take precedence (override=False)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

OLLAMA_BASE_URL = "http://localhost:11434"

# ── Models under comparison ─────────────────────────────────
# Ollama model tags + metadata for display and analysis
MODELS = {
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
# Embedding model (must be pulled separately: ollama pull nomic-embed-text)
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768          # nomic-embed-text output dim

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
}

# ── Privacy settings ─────────────────────────────────────────
PRIVACY_CONFIG = {
    "network_calls_blocked": True,   # all inference is local
    "data_leaves_device": False,
    "logs_sent_to_vendor": False,
    "gdpr_compliant": True,          # no external data transfer
    "telemetry": False,              # chromadb telemetry disabled
}
