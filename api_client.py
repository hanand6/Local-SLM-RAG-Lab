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
from config import (
    API_MAX_RETRIES, API_BACKOFF_BASE_S, API_BACKOFF_MAX_S,
    API_TIMEOUT_S, API_MAX_OUTPUT_TOKENS,
)


# ── Error helpers ────────────────────────────────────────────

def _status_code(exc: Exception) -> Optional[int]:
    return getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )


def _retry_after_s(exc: Exception) -> Optional[float]:
    """Read the Retry-After header from an API error, if present."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    try:
        val = headers.get("retry-after") or headers.get("Retry-After")
        return float(val) if val else None
    except (TypeError, ValueError):
        return None


def friendly_api_error(exc: Exception, model: str) -> str:
    """Turn raw API exceptions into a message that says what to do."""
    code = _status_code(exc)
    if code == 401:
        return "Invalid or missing API key (401). Check your key in .env."
    if code == 402:
        return "Out of credits (402). Pick a ':free' model or add credits."
    if code == 403:
        return (
            f"Access to '{model}' is restricted (403). Some free models are limited "
            "to specific apps. Pick another model via OPENROUTER_MODELS in .env."
        )
    if code == 404:
        return (
            f"Model '{model}' was not found (404). Free models are renamed or removed "
            "often. See the Setup tab for models available now, then update "
            "OPENROUTER_MODELS in .env."
        )
    if code == 429:
        return (
            "Rate limited (429) even after retrying. Free models allow roughly "
            "20 requests/min and a daily cap. Wait a minute, run fewer calls, or "
            "pick another model."
        )
    return str(exc)


# ── OpenAI-compatible client (OpenRouter, NVIDIA NIM, ...) ───

class OpenAICompatClient:
    """
    Wraps any OpenAI-compatible API endpoint (OpenRouter, NVIDIA NIM, etc.)
    Returns BenchmarkResult so it plugs into the existing benchmark pipeline.

    Retries on 429 (rate limit) and 5xx errors with exponential backoff.
    """

    RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, api_key: str, base_url: str, model: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        self.model = model
        # max_retries=0: we do our own retry loop so we can surface it clearly.
        self._client = OpenAI(
            api_key=api_key, base_url=base_url,
            max_retries=0, timeout=API_TIMEOUT_S,
        )
        self._base_url = base_url

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = API_MAX_OUTPUT_TOKENS,
        run_index: int = 0,
    ) -> BenchmarkResult:
        """
        Call the API (with rate-limit retries) and return a BenchmarkResult.

        Reasoning models sometimes spend their whole output budget thinking and
        return no visible answer. When that happens we retry once with double
        the budget and (on OpenRouter) low reasoning effort.
        """
        result = BenchmarkResult(
            model=self.model,
            run_index=run_index,
            prompt_preview=prompt[:120],
        )

        self._generate_with_retries(result, prompt, system, temperature, max_tokens)
        if result.error or result.response.strip():
            return result

        # ── Empty answer: one recovery attempt ──
        first_reasoning = result.reasoning_tokens
        retry_budget = max_tokens * 2
        extra = {"reasoning": {"effort": "low"}} if "openrouter.ai" in self._base_url else None
        self._generate_with_retries(result, prompt, system, temperature, retry_budget, extra)
        result.empty_answer_retry = True
        if result.error or result.response.strip():
            return result

        result.error = (
            f"Empty answer from '{self.model}', even after a retry with a "
            f"{retry_budget}-token budget and low reasoning effort "
            f"(it used {max(first_reasoning, result.reasoning_tokens)} tokens thinking). "
            "This model struggles with this prompt — try another model, or raise "
            "API_MAX_OUTPUT_TOKENS in config.py."
            if first_reasoning or result.reasoning_tokens or result.finish_reason == "length"
            else f"Empty answer from '{self.model}' (twice). The provider returned no "
            "text; try again later or pick another model."
        )
        return result

    def _generate_with_retries(
        self,
        result: BenchmarkResult,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        extra_body: Optional[dict] = None,
    ) -> None:
        """Retry on 429/5xx with exponential backoff. Sets result.error on failure."""
        last_exc: Optional[Exception] = None
        for attempt in range(API_MAX_RETRIES + 1):
            try:
                self._generate_once(result, prompt, system, temperature, max_tokens, extra_body)
                result.error = ""
                result.retries += attempt
                return
            except Exception as e:
                last_exc = e
                code = _status_code(e)
                if code not in self.RETRY_STATUS or attempt == API_MAX_RETRIES:
                    break
                wait = _retry_after_s(e) or API_BACKOFF_BASE_S * (2 ** attempt)
                time.sleep(min(wait, API_BACKOFF_MAX_S))

        result.error = friendly_api_error(last_exc, self.model)
        result.retries += attempt

    def _generate_once(
        self,
        result: BenchmarkResult,
        prompt: str,
        system: str,
        temperature: float,
        max_tokens: int,
        extra_body: Optional[dict] = None,
    ) -> None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        wall_start = time.perf_counter()
        first_token_time: Optional[float] = None    # first token of any kind
        first_answer_time: Optional[float] = None   # first visible answer token
        collected_text = []
        reasoning_chars = 0
        finish_reason = ""
        usage = None

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            **({"extra_body": extra_body} if extra_body else {}),
        )

        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            delta = choice.delta
            if not delta:
                continue
            # Reasoning models (via OpenRouter) stream hidden thinking in a
            # separate `reasoning` field before the visible answer.
            reasoning = getattr(delta, "reasoning", None) or (
                (getattr(delta, "model_extra", None) or {}).get("reasoning")
            )
            now = time.perf_counter()
            if reasoning:
                reasoning_chars += len(reasoning)
                first_token_time = first_token_time or now
            if delta.content:
                first_token_time = first_token_time or now
                first_answer_time = first_answer_time or now
                collected_text.append(delta.content)

        wall_s = time.perf_counter() - wall_start

        result.response = "".join(collected_text)
        result.finish_reason = finish_reason
        result.wall_clock_s = wall_s
        result.total_duration_ms = wall_s * 1000

        # TTFT = wall time to the first streamed token, reasoning included
        # (network + provider queue + prefill). Generation = the rest.
        result.ttft_ms = (
            (first_token_time - wall_start) * 1000
            if first_token_time else wall_s * 1000
        )
        result.eval_duration_ms = result.total_duration_ms - result.ttft_ms
        # Time until the user sees the first word of the answer.
        result.time_to_answer_ms = (
            (first_answer_time - wall_start) * 1000
            if first_answer_time else wall_s * 1000
        )

        # Exact token counts when the API reports usage; otherwise ~4 chars/token.
        if usage is not None and getattr(usage, "completion_tokens", None):
            result.output_tokens = usage.completion_tokens
            result.prompt_tokens = usage.prompt_tokens or max(1, len(prompt) // 4)
            details = getattr(usage, "completion_tokens_details", None)
            result.reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
        else:
            result.output_tokens = max(1, (len(result.response) + reasoning_chars) // 4)
            result.prompt_tokens = max(1, len(prompt) // 4)
            result.reasoning_tokens = reasoning_chars // 4

        if result.eval_duration_ms > 0 and result.output_tokens > 0:
            result.tokens_per_second = result.output_tokens / (
                result.eval_duration_ms / 1000
            )


# Backwards-compatible name used elsewhere in the project.
NvidiaApiClient = OpenAICompatClient


def list_openrouter_free_models(timeout: float = 10.0) -> list[dict]:
    """
    Public OpenRouter model catalogue, filtered to free models.
    No API key needed. Returns [{id, name, context_length}, ...].
    """
    import requests
    from config import OPENROUTER_BASE_URL

    r = requests.get(f"{OPENROUTER_BASE_URL}/models", timeout=timeout)
    r.raise_for_status()
    return sorted(
        (
            {
                "id": m["id"],
                "name": m.get("name", m["id"]),
                "context_length": m.get("context_length", 0),
            }
            for m in r.json().get("data", [])
            if m.get("id", "").endswith(":free")
        ),
        key=lambda m: m["id"],
    )


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
