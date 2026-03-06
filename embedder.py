# ============================================================
# embedder.py — Unified embedding client
#
# Supports:
#   - nomic-embed-text  → Ollama local API (768-dim)
#   - all-MiniLM-L6-v2 → SentenceTransformers / HuggingFace (384-dim)
#
# Privacy: nomic-embed-text is 100% local.
#          all-MiniLM-L6-v2 downloads model weights from HuggingFace
#          on first use (~90 MB), then runs fully offline.
# ============================================================

import requests
from config import OLLAMA_BASE_URL


# ── Embedding model registry ─────────────────────────────────

EMBEDDING_OPTIONS: dict[str, dict] = {
    "nomic-embed-text": {
        "label": "nomic-embed-text (Ollama · 768-dim)",
        "dim": 768,
        "backend": "ollama",
        "note": "Pull with: ollama pull nomic-embed-text",
    },
    "all-MiniLM-L6-v2": {
        "label": "all-MiniLM-L6-v2 (SentenceTransformers · 384-dim)",
        "dim": 384,
        "backend": "sentence_transformers",
        "note": "Auto-downloaded from HuggingFace on first use (~90 MB).",
    },
}


# ── Embedding client ─────────────────────────────────────────

class EmbeddingClient:
    """
    Unified embedding interface.
    Lazy-loads the SentenceTransformer model on first call.
    """

    def __init__(self, model: str = "nomic-embed-text"):
        if model not in EMBEDDING_OPTIONS:
            raise ValueError(
                f"Unknown embedding model '{model}'. "
                f"Valid options: {list(EMBEDDING_OPTIONS)}"
            )
        self.model = model
        self._meta = EMBEDDING_OPTIONS[model]
        self._st_model = None   # lazy-loaded

    # ── Public API ───────────────────────────────────────────

    @property
    def dim(self) -> int:
        return self._meta["dim"]

    @property
    def backend(self) -> str:
        return self._meta["backend"]

    def embed(self, text: str) -> list[float]:
        if self._meta["backend"] == "ollama":
            return self._ollama_embed(text)
        return self._st_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed — more efficient for SentenceTransformers."""
        if self._meta["backend"] == "sentence_transformers":
            self._ensure_st_loaded()
            return self._st_model.encode(
                texts, show_progress_bar=False, batch_size=32
            ).tolist()
        return [self.embed(t) for t in texts]

    # ── Backends ────────────────────────────────────────────

    def _ollama_embed(self, text: str) -> list[float]:
        payload = {"model": self.model, "prompt": text}
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["embedding"]

    def _st_embed(self, text: str) -> list[float]:
        self._ensure_st_loaded()
        return self._st_model.encode(text, show_progress_bar=False).tolist()

    def _ensure_st_loaded(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model)
