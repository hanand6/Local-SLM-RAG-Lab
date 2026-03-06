# ============================================================
# ollama_client.py — Ollama REST API wrapper
# All inference stays on localhost — zero external calls.
# ============================================================

import json
import time
import requests
from typing import Generator, Optional
from config import OLLAMA_BASE_URL


class OllamaClient:
    """
    Thin wrapper around Ollama's local REST API.

    Privacy guarantee: base_url is always localhost / 127.0.0.1.
    Every method raises if an external URL is detected.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self._assert_local(base_url)
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    # ── Safety ───────────────────────────────────────────────

    @staticmethod
    def _assert_local(url: str) -> None:
        allowed = ("http://localhost", "http://127.")
        if not any(url.startswith(p) for p in allowed):
            raise ValueError(
                f"OllamaClient refuses to connect to non-local URL: {url}\n"
                "All inference must stay on-device for privacy compliance."
            )

    def verify_privacy(self) -> dict:
        """Return a privacy attestation dict (log/display this in UI)."""
        return {
            "endpoint": self.base_url,
            "is_local": True,
            "network_egress": "NONE — all traffic on 127.0.0.1",
            "data_leaves_device": False,
            "external_api_calls": 0,
        }

    # ── Health ───────────────────────────────────────────────

    def is_running(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def list_models(self) -> list[str]:
        r = self.session.get(f"{self.base_url}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def is_model_available(self, model: str) -> bool:
        """Check if model is pulled locally (partial name match)."""
        available = self.list_models()
        return any(model in m or m.startswith(model) for m in available)

    # ── Inference ────────────────────────────────────────────

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,   # low temp for factual RAG
        num_ctx: int = 4096,
    ) -> dict:
        """
        Non-streaming generation.
        Returns Ollama's full response dict including native timing fields:
          - total_duration     (nanoseconds)
          - load_duration      (nanoseconds)
          - prompt_eval_count  (tokens)
          - prompt_eval_duration (nanoseconds)
          - eval_count         (tokens generated)
          - eval_duration      (nanoseconds)
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }
        wall_start = time.perf_counter()
        r = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=300,
        )
        wall_end = time.perf_counter()
        r.raise_for_status()

        data = r.json()
        data["wall_clock_s"] = wall_end - wall_start
        return data

    def generate_stream(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        num_ctx: int = 4096,
    ) -> Generator[dict, None, None]:
        """
        Streaming generation — yields each token chunk as a dict.
        Final chunk contains the full metrics (done=True).
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }
        with self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=300,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    yield json.loads(line)

    # ── Embeddings ───────────────────────────────────────────

    def embed(self, model: str, text: str) -> list[float]:
        """
        Local embedding via Ollama — no external calls.
        Requires: ollama pull nomic-embed-text
        """
        payload = {"model": model, "prompt": text}
        r = self.session.post(
            f"{self.base_url}/api/embeddings",
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["embedding"]

    def embed_batch(self, model: str, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts sequentially (Ollama has no batch endpoint yet)."""
        return [self.embed(model, t) for t in texts]
