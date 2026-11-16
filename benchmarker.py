# ============================================================
# benchmarker.py — Inference performance measurement
#
# Metrics captured per run:
#   - TTFT  : Time-to-first-token (ms)  — latency felt by user
#   - TPS   : Tokens per second         — generation throughput
#   - P50/P95 latency across N runs
#   - RAM delta during inference
#   - Model load time (cold vs warm)
# ============================================================

import os
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import psutil

from ollama_client import OllamaClient


# ── Data model ───────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    model: str
    run_index: int
    prompt_preview: str       # first 120 chars

    # Raw Ollama timing (converted to ms)
    total_duration_ms: float = 0.0
    load_duration_ms: float = 0.0          # >0 on cold start
    prompt_eval_duration_ms: float = 0.0   # tokenisation + prefill
    eval_duration_ms: float = 0.0          # actual generation

    # Token counts
    prompt_tokens: int = 0
    output_tokens: int = 0

    # Derived
    tokens_per_second: float = 0.0
    ttft_ms: float = 0.0                   # approx: prompt_eval + load
    time_per_output_token_ms: float = 0.0

    # System
    ram_delta_mb: float = 0.0
    wall_clock_s: float = 0.0

    # Full response (for quality comparison)
    response: str = ""
    error: str = ""

    # Cloud APIs: how many times we retried after a rate limit / server error
    retries: int = 0
    # Reasoning models: hidden "thinking" tokens (included in output_tokens)
    reasoning_tokens: int = 0
    # Time until the first visible answer token (after any reasoning)
    time_to_answer_ms: float = 0.0
    finish_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_cold_start(self) -> bool:
        return self.load_duration_ms > 200  # >200ms load = model was swapped out


# ── Benchmarker ──────────────────────────────────────────────

class Benchmarker:
    """
    Runs structured inference benchmarks against local Ollama models.

    Usage:
        bm = Benchmarker()
        result = bm.run_single("llama3.2:3b", prompt="...")
        suite  = bm.run_suite(models=[...], prompts=[...])
        stats  = bm.summary_stats(suite)
    """

    def __init__(self, base_url: str = None):
        kwargs = {"base_url": base_url} if base_url else {}
        self.client = OllamaClient(**kwargs)

    # ── Single run ───────────────────────────────────────────

    def run_single(
        self,
        model: str,
        prompt: str,
        system: str = "",
        run_index: int = 0,
        temperature: float = 0.1,
    ) -> BenchmarkResult:
        """Run one inference pass and return a rich BenchmarkResult."""
        ram_before = psutil.virtual_memory().used / (1024 ** 2)

        result = BenchmarkResult(
            model=model,
            run_index=run_index,
            prompt_preview=prompt[:120],
        )

        try:
            data = self.client.generate(
                model=model,
                prompt=prompt,
                system=system,
                temperature=temperature,
            )
        except Exception as e:
            result.error = str(e)
            return result

        ram_after = psutil.virtual_memory().used / (1024 ** 2)

        # ── Populate from Ollama native fields (nanoseconds → ms) ──
        ns = 1_000_000  # 1ms in ns

        result.total_duration_ms       = data.get("total_duration", 0) / ns
        result.load_duration_ms        = data.get("load_duration", 0) / ns
        result.prompt_eval_duration_ms = data.get("prompt_eval_duration", 0) / ns
        result.eval_duration_ms        = data.get("eval_duration", 0) / ns
        result.prompt_tokens           = data.get("prompt_eval_count", 0)
        result.output_tokens           = data.get("eval_count", 0)
        result.response                = data.get("response", "")
        result.wall_clock_s            = data.get("wall_clock_s", 0)
        result.ram_delta_mb            = ram_after - ram_before

        # ── Derived metrics ──────────────────────────────────
        if result.eval_duration_ms > 0 and result.output_tokens > 0:
            result.tokens_per_second = result.output_tokens / (
                result.eval_duration_ms / 1000
            )
            result.time_per_output_token_ms = result.eval_duration_ms / result.output_tokens

        # TTFT = time before first generated token
        # = model load (if cold) + prompt tokenisation/prefill
        result.ttft_ms = result.prompt_eval_duration_ms + (
            result.load_duration_ms if result.is_cold_start else 0
        )

        return result

    # ── Suite runner ─────────────────────────────────────────

    def run_suite(
        self,
        models: list[str],
        prompts: list[str],
        system: str = "",
        n_repeats: int = 1,
        progress_cb=None,         # optional callback(model, prompt_i, run_i)
    ) -> list[BenchmarkResult]:
        """
        Run every model × every prompt × n_repeats.
        Returns flat list of BenchmarkResult objects.
        """
        results = []
        total = len(models) * len(prompts) * n_repeats
        done = 0

        for model in models:
            for p_idx, prompt in enumerate(prompts):
                for r_idx in range(n_repeats):
                    done += 1
                    if progress_cb:
                        progress_cb(model, p_idx, r_idx, done, total)
                    res = self.run_single(
                        model=model,
                        prompt=prompt,
                        system=system,
                        run_index=r_idx,
                    )
                    results.append(res)

        return results

    # ── Statistics ───────────────────────────────────────────

    def summary_stats(self, results: list[BenchmarkResult]) -> dict:
        """
        Per-model aggregated statistics.

        Returns:
            {
              "llama3.2:3b": {
                "p50_latency_ms": ...,
                "p95_latency_ms": ...,
                "avg_tps": ...,
                "avg_ttft_ms": ...,
                "avg_output_tokens": ...,
                "cold_starts": ...,
                "errors": ...,
              },
              ...
            }
        """
        by_model = defaultdict(list)
        for r in results:
            by_model[r.model].append(r)

        summary = {}
        for model, runs in by_model.items():
            good = [r for r in runs if not r.error]
            if not good:
                summary[model] = {"error": "all runs failed"}
                continue

            latencies = sorted(r.total_duration_ms for r in good)
            tps_list  = [r.tokens_per_second for r in good if r.tokens_per_second > 0]
            ttft_list = [r.ttft_ms for r in good]

            n = len(latencies)
            p95_idx = max(0, int(n * 0.95) - 1)

            summary[model] = {
                "runs":               n,
                "errors":             len([r for r in runs if r.error]),
                "cold_starts":        sum(1 for r in good if r.is_cold_start),
                "p50_latency_ms":     round(statistics.median(latencies), 1),
                "p95_latency_ms":     round(latencies[p95_idx], 1),
                "min_latency_ms":     round(min(latencies), 1),
                "max_latency_ms":     round(max(latencies), 1),
                "avg_tps":            round(statistics.mean(tps_list), 1) if tps_list else 0,
                "avg_ttft_ms":        round(statistics.mean(ttft_list), 1) if ttft_list else 0,
                "avg_output_tokens":  round(statistics.mean(r.output_tokens for r in good), 1),
                "avg_prompt_tokens":  round(statistics.mean(r.prompt_tokens for r in good), 1),
            }

        return summary

    # ── Cost calculator ──────────────────────────────────────

    @staticmethod
    def api_cost_equivalent(
        prompt_tokens: int,
        output_tokens: int,
        provider: str = "GPT-4o (OpenAI)",
    ) -> float:
        """
        What would this inference cost if done via a cloud API?
        Returns USD cost (to show the $0 local advantage).
        """
        from config import API_COST_REFERENCE
        pricing = API_COST_REFERENCE.get(provider, {"input": 2.50, "output": 10.00})
        cost = (prompt_tokens / 1_000_000 * pricing["input"]) + \
               (output_tokens / 1_000_000 * pricing["output"])
        return round(cost, 6)
