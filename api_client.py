# ============================================================
# api_client.py — Cloud API client (OpenAI-compatible + Gemini)
#
# Supports:
#   - NVIDIA NIM (Kimi K2 Instruct, OpenAI-compatible)
#   - Google Gemini (via google-generativeai SDK)
#
# Returns BenchmarkResult objects so cloud models slot into
# the same benchmark/compare pipeline as local Ollama models.
#
# Privacy note: these clients make EXTERNAL network calls.
# Data leaves the device. Not suitable for confidential docs.
# ============================================================

import time
from typing import Optional

from benchmarker import BenchmarkResult


# ── NVIDIA NIM / OpenAI-compatible client ────────────────────

class NvidiaApiClient:
    """
    Wraps any OpenAI-compatible API endpoint (NVIDIA NIM, OpenRouter, etc.)
    Returns BenchmarkResult so it plugs into the existing benchmark pipeline.
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        run_index: int = 0,
    ) -> BenchmarkResult:
        """Call the API and return a BenchmarkResult with timing."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        result = BenchmarkResult(
            model=self.model,
            run_index=run_index,
            prompt_preview=prompt[:120],
        )

        try:
            wall_start = time.perf_counter()
            first_token_time: Optional[float] = None
            collected_text = []

            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    collected_text.append(delta.content)

            wall_end = time.perf_counter()
            wall_s = wall_end - wall_start

            result.response = "".join(collected_text)
            result.wall_clock_s = wall_s
            result.total_duration_ms = wall_s * 1000

            # TTFT = wall time to first token
            result.ttft_ms = (
                (first_token_time - wall_start) * 1000
                if first_token_time else wall_s * 1000
            )

            # Generation time = total - ttft
            result.eval_duration_ms = result.total_duration_ms - result.ttft_ms

            # Approximate token counts from character count
            # (cloud APIs don't always return exact counts in streaming)
            result.output_tokens = max(1, len(result.response) // 4)
            result.prompt_tokens = max(1, len(prompt) // 4)

            if result.eval_duration_ms > 0 and result.output_tokens > 0:
                result.tokens_per_second = result.output_tokens / (
                    result.eval_duration_ms / 1000
                )

        except Exception as e:
            result.error = str(e)

        return result


# ── Gemini client ─────────────────────────────────────────────

class GeminiApiClient:
    """
    Wraps Google Gemini API via google-generativeai SDK.
    Returns BenchmarkResult for unified benchmarking.
    """

    def __init__(self, api_key: str, model: str):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Install google-generativeai: pip install google-generativeai"
            )
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model_name = model
        self._genai = genai

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        run_index: int = 0,
    ) -> BenchmarkResult:
        """Call Gemini API and return a BenchmarkResult with timing."""
        result = BenchmarkResult(
            model=self.model_name,
            run_index=run_index,
            prompt_preview=prompt[:120],
        )

        try:
            generation_config = self._genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            # System instruction support (Gemini 1.5+ / 2.0+)
            model_kwargs = {"generation_config": generation_config}
            if system:
                model_kwargs["system_instruction"] = system

            model = self._genai.GenerativeModel(self.model_name, **model_kwargs)

            wall_start = time.perf_counter()
            first_token_time: Optional[float] = None
            collected_text = []

            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    collected_text.append(chunk.text)

            wall_end = time.perf_counter()
            wall_s = wall_end - wall_start

            result.response = "".join(collected_text)
            result.wall_clock_s = wall_s
            result.total_duration_ms = wall_s * 1000
            result.ttft_ms = (
                (first_token_time - wall_start) * 1000
                if first_token_time else wall_s * 1000
            )
            result.eval_duration_ms = result.total_duration_ms - result.ttft_ms
            result.output_tokens = max(1, len(result.response) // 4)
            result.prompt_tokens = max(1, len(prompt) // 4)

            if result.eval_duration_ms > 0 and result.output_tokens > 0:
                result.tokens_per_second = result.output_tokens / (
                    result.eval_duration_ms / 1000
                )

        except Exception as e:
            result.error = str(e)

        return result
