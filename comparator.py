# ============================================================
# comparator.py — Run the same question across all 4 models
# and produce quality-vs-speed analysis.
# ============================================================

import statistics
from typing import Optional

import pandas as pd

from config import MODELS, API_COST_REFERENCE
from benchmarker import Benchmarker
from rag_engine import RAGEngine


class ModelComparator:
    """
    Runs identical prompts across multiple models and
    surfaces quality-vs-speed-vs-cost tradeoffs.
    """

    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine

    # ── Core comparison ──────────────────────────────────────

    def compare(
        self,
        question: str,
        models: list[str],
        top_k: int = 5,
        progress_cb=None,
    ) -> list[dict]:
        """
        Run the same RAG question against each model.
        Returns list of result dicts (one per model).
        """
        results = []
        for i, model in enumerate(models):
            if progress_cb:
                progress_cb(model, i + 1, len(models))
            result = self.rag.query(question, model, top_k=top_k)
            results.append(result)
        return results

    # ── DataFrame views ──────────────────────────────────────

    def to_answers_df(self, results: list[dict]) -> pd.DataFrame:
        """Side-by-side answer comparison table."""
        rows = []
        for r in results:
            label = MODELS.get(r["model"], {}).get("label", r["model"])
            rows.append({
                "Model": label,
                "Answer": r["answer"],
                "Tokens/sec": r["tokens_per_sec"],
                "Total (ms)": r["total_ms"],
                "Output tokens": r["output_tokens"],
            })
        return pd.DataFrame(rows)

    def to_metrics_df(self, results: list[dict]) -> pd.DataFrame:
        """Detailed performance metrics table."""
        rows = []
        for r in results:
            label  = MODELS.get(r["model"], {}).get("label", r["model"])
            family = MODELS.get(r["model"], {}).get("family", "—")
            params = MODELS.get(r["model"], {}).get("params_b", "?")
            disk   = MODELS.get(r["model"], {}).get("disk_size_gb", "?")

            br = r.get("benchmark_result")
            prompt_tokens  = br.prompt_tokens if br else 0
            output_tokens  = r["output_tokens"]
            cost_gpt4o     = Benchmarker.api_cost_equivalent(
                prompt_tokens, output_tokens, "GPT-4o (OpenAI)"
            )

            rows.append({
                "Model":              label,
                "Family":             family,
                "Params (B)":         params,
                "Disk (GB)":          disk,
                "Retrieval (ms)":     r["retrieval_ms"],
                "TTFT (ms)":          r["ttft_ms"],
                "Generation (ms)":    r["generation_ms"],
                "Total latency (ms)": r["total_ms"],
                "Tokens/sec":         r["tokens_per_sec"],
                "Output tokens":      output_tokens,
                "Prompt tokens":      prompt_tokens,
                "API equiv. cost $":  f"${cost_gpt4o:.5f}",
                "Local cost $":       "$0.00000",
                "Privacy":            _privacy_label(r["model"]),
            })
        return pd.DataFrame(rows)

    def to_tradeoff_df(self, results: list[dict]) -> pd.DataFrame:
        """
        Quality-vs-Speed tradeoff scoring.

        Speed score  (0–5): based on tokens/sec (50 t/s = 5.0)
        Latency score(0–5): lower total latency → higher score
        Answer length score: proxy for response completeness
        """
        rows = []
        all_tps      = [r["tokens_per_sec"] for r in results if r["tokens_per_sec"] > 0]
        all_latency  = [r["total_ms"] for r in results if r["total_ms"] > 0]
        all_tokens   = [r["output_tokens"] for r in results if r["output_tokens"] > 0]

        max_tps     = max(all_tps)     if all_tps     else 1
        max_lat     = max(all_latency) if all_latency else 1
        max_tokens  = max(all_tokens)  if all_tokens  else 1

        for r in results:
            label  = MODELS.get(r["model"], {}).get("label", r["model"])
            params = MODELS.get(r["model"], {}).get("params_b", 0)

            tps        = r["tokens_per_sec"]
            latency    = r["total_ms"]
            out_tokens = r["output_tokens"]

            speed_score   = round((tps / max_tps) * 5, 1) if max_tps else 0
            latency_score = round((1 - latency / max_lat) * 5, 1) if max_lat else 0
            depth_score   = round((out_tokens / max_tokens) * 5, 1) if max_tokens else 0

            overall = round(
                (speed_score * 0.35) + (latency_score * 0.35) + (depth_score * 0.30),
                1
            )

            rows.append({
                "Model":              label,
                "Params (B)":         params,
                "Speed (/5)":         speed_score,
                "Latency (/5)":       latency_score,
                "Answer depth (/5)":  depth_score,
                "Overall score (/5)": overall,
                "Tokens/sec":         tps,
                "Total latency (ms)": latency,
                "Best for":           _best_use_case(speed_score, latency_score, depth_score),
                "Cost (local)":       "$0",
                "Cost (GPT-4o equiv)": _cost_label(
                    r.get("benchmark_result").prompt_tokens if r.get("benchmark_result") else 0,
                    out_tokens
                ),
            })

        return pd.DataFrame(rows).sort_values("Overall score (/5)", ascending=False)

    # ── Constraint analysis ──────────────────────────────────

    @staticmethod
    def constraint_analysis(results: list[dict]) -> dict:
        """
        Evaluates each model against real-world constraints:
          - Privacy: always pass (all local)
          - Latency SLA: <3000ms for interactive, <10000ms for batch
          - Cost: always $0 (vs. API)
          - Memory: approximate from model size
        """
        analysis = {}
        for r in results:
            model  = r["model"]
            label  = MODELS.get(model, {}).get("label", model)
            params = MODELS.get(model, {}).get("params_b", 0)
            total  = r["total_ms"]

            is_local = MODELS.get(model, {}).get("data_local", True)

            # Rule-of-thumb RAM: ~0.6 GB per billion params at Q4 (local only)
            est_ram_gb = params * 0.6 if is_local else 0.0

            analysis[label] = {
                "privacy_ok":          is_local,
                "interactive_sla_ok":  total < 3000,   # <3s for chat
                "batch_sla_ok":        total < 10000,  # <10s for doc processing
                "cost_per_query":      "$0.00",        # local or free-tier cloud
                "est_ram_gb":          round(est_ram_gb, 1),
                "gdpr_compliant":      is_local,
                "data_leaves_device":  not is_local,
            }
        return analysis


# ── Helpers ──────────────────────────────────────────────────

def _privacy_label(model: str) -> str:
    if MODELS.get(model, {}).get("data_local", True):
        return "✅ 100% local"
    return "☁️ Cloud — data leaves device"


def _best_use_case(speed: float, latency: float, depth: float) -> str:
    if speed >= 4.0 and latency >= 4.0:
        return "Real-time chat / quick Q&A"
    elif depth >= 4.0:
        return "Deep document analysis"
    elif speed >= 3.5:
        return "Bulk document processing"
    else:
        return "Quality-first offline analysis"


def _cost_label(prompt_tokens: int, output_tokens: int) -> str:
    cost = Benchmarker.api_cost_equivalent(prompt_tokens, output_tokens)
    return f"${cost:.4f}" if cost > 0 else "$0.0000"
