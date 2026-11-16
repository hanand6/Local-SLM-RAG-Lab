# ============================================================
# ragas_evaluator.py — Multi-judge RAGAS evaluation wrapper
#
# Supported judges:
#   "gemini"       → Gemini 2.5 Flash (cloud, via langchain-google-genai)
#   "qwen72b"      → Qwen2.5-72B-Instruct (NVIDIA NIM, via langchain-openai)
#   "qwen3:4b"     → Local Ollama model (via langchain-community)
#   "llama3.2"     → Local Ollama model (via langchain-community)
#
# Metrics (reference-free):
#   faithfulness, answer_relevancy, context_precision
#
# Metrics (reference-based, requires ground_truth):
#   + context_recall, answer_correctness, answer_similarity
# ============================================================

from __future__ import annotations

import time
import warnings
from typing import Optional

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="ragas")

# ── Judge configs ─────────────────────────────────────────────

JUDGE_OPTIONS: dict[str, dict] = {
    "openrouter": {
        "label":   "🌐 OpenRouter (free)",
        "backend": "openrouter",
        "warning": "Uses OPENROUTER_JUDGE_MODEL. Each eval makes several calls — watch free-tier limits.",
    },
    "gemini": {
        "label":   "☁️ Gemini 2.5 Flash",
        "backend": "gemini",
        "warning": None,
    },
    "qwen72b": {
        "label":   "☁️ Qwen2.5-72B (NVIDIA NIM)",
        "backend": "nvidia_nim",
        "warning": "Rate-limited on free tier (~5 RPM). Batch eval may be slow.",
    },
    "qwen3:4b": {
        "label":   "🖥️ qwen3:4b (Ollama)",
        "backend": "ollama",
        "warning": None,
    },
    "llama3.2:latest": {
        "label":   "🖥️ llama3.2:3b (Ollama)",
        "backend": "ollama",
        "warning": "Small model — may produce NaN scores due to JSON formatting failures.",
    },
}


# ── LLM factory ──────────────────────────────────────────────

def _build_llm(judge_key: str):
    """Return a LangChain LLM wrapper for the given judge key."""
    from config import (
        GEMINI_API_KEY, NVIDIA_API_KEY, NVIDIA_BASE_URL,
        OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_JUDGE_MODEL,
        API_MAX_RETRIES,
    )

    cfg = JUDGE_OPTIONS[judge_key]
    backend = cfg["backend"]

    if backend == "openrouter":
        from langchain_openai import ChatOpenAI
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set in .env")
        return ChatOpenAI(
            model=OPENROUTER_JUDGE_MODEL,
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            temperature=0,
            max_retries=API_MAX_RETRIES,
        )

    elif backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0,
        )

    elif backend == "nvidia_nim":
        from langchain_openai import ChatOpenAI
        if not NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY not set in .env")
        return ChatOpenAI(
            model="qwen/qwen2.5-72b-instruct",
            base_url=NVIDIA_BASE_URL,
            api_key=NVIDIA_API_KEY,
            temperature=0,
            max_retries=3,
        )

    elif backend == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=judge_key,
            temperature=0,
            format="json",          # request JSON mode where supported
            num_predict=512,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")


def _build_embeddings():
    """Return a LangChain embeddings wrapper using our EmbeddingClient."""
    from langchain_core.embeddings import Embeddings
    from embedder import EmbeddingClient
    from config import EMBEDDING_MODEL

    class _LocalEmbeddings(Embeddings):
        def __init__(self):
            self._client = EmbeddingClient(EMBEDDING_MODEL)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._client.embed_batch(texts)

        def embed_query(self, text: str) -> list[float]:
            return self._client.embed(text)

    return _LocalEmbeddings()


# ── Metric builder ────────────────────────────────────────────

def _build_metrics(llm, embeddings, has_ground_truth: bool):
    """Return the RAGAS metric objects for this run."""
    try:
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            LLMContextPrecisionWithoutReference,
        )
    except ImportError:
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            LLMContextPrecisionWithoutReference,
        )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    llm_w   = LangchainLLMWrapper(llm)
    embed_w = LangchainEmbeddingsWrapper(embeddings)

    metrics = [
        Faithfulness(llm=llm_w),
        AnswerRelevancy(llm=llm_w, embeddings=embed_w),
        LLMContextPrecisionWithoutReference(llm=llm_w),
    ]

    if has_ground_truth:
        from ragas.metrics import (
            ContextRecall,
            AnswerCorrectness,
            SemanticSimilarity,
        )
        metrics += [
            ContextRecall(llm=llm_w),
            AnswerCorrectness(llm=llm_w),
            SemanticSimilarity(embeddings=embed_w),
        ]

    return metrics


# ── Public API ────────────────────────────────────────────────

class RagasEvaluator:
    """
    Evaluate RAG outputs with RAGAS using one or more judge LLMs.

    Usage:
        evaluator = RagasEvaluator()

        # Single query (reference-free)
        scores = evaluator.evaluate_single(
            judge="gemini",
            question="What is the capital?",
            answer="The capital is Paris.",
            contexts=["France's capital is Paris."],
        )

        # Batch with ground truth
        df = evaluator.evaluate_dataset(
            judges=["gemini", "qwen3:4b"],
            rows=[
                {"question": "...", "answer": "...", "contexts": [...], "ground_truth": "..."},
            ]
        )
    """

    def evaluate_single(
        self,
        judge: str,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Evaluate a single RAG result.
        Returns dict of {metric_name: score} (NaN on failure).
        """
        from datasets import Dataset
        from ragas import evaluate

        row = {
            "question":  [question],
            "answer":    [answer],
            "contexts":  [contexts],
        }
        if ground_truth:
            row["ground_truth"] = [ground_truth]

        ds = Dataset.from_dict(row)
        llm  = _build_llm(judge)
        emb  = _build_embeddings()
        mets = _build_metrics(llm, emb, has_ground_truth=bool(ground_truth))

        try:
            t0     = time.perf_counter()
            result = evaluate(dataset=ds, metrics=mets, raise_exceptions=False)
            elapsed = round(time.perf_counter() - t0, 1)
            scores  = {k: round(float(v), 3) for k, v in result.items()}
            scores["eval_time_s"] = elapsed
        except Exception as e:
            scores = {m.name: float("nan") for m in mets}
            scores["error"] = str(e)

        return scores

    def evaluate_dataset(
        self,
        judges: list[str],
        rows: list[dict],
    ) -> pd.DataFrame:
        """
        Evaluate multiple rows with multiple judges.
        Returns a DataFrame: one row per input row, columns = metric per judge.
        """
        from datasets import Dataset
        from ragas import evaluate

        has_gt = any("ground_truth" in r for r in rows)

        data = {
            "question": [r["question"] for r in rows],
            "answer":   [r["answer"]   for r in rows],
            "contexts": [r["contexts"] for r in rows],
        }
        if has_gt:
            data["ground_truth"] = [r.get("ground_truth", "") for r in rows]

        ds = Dataset.from_dict(data)

        all_scores: dict[str, list] = {
            "question": [r["question"] for r in rows],
        }

        for judge in judges:
            llm  = _build_llm(judge)
            emb  = _build_embeddings()
            mets = _build_metrics(llm, emb, has_ground_truth=has_gt)
            tag  = JUDGE_OPTIONS[judge]["label"].split(" ", 1)[-1].strip("️").strip()

            try:
                result = evaluate(dataset=ds, metrics=mets, raise_exceptions=False)
                result_df = result.to_pandas()
                for metric in mets:
                    col_name = f"{metric.name} [{tag}]"
                    vals = result_df[metric.name].tolist() if metric.name in result_df else [float("nan")] * len(rows)
                    all_scores[col_name] = [round(float(v), 3) for v in vals]
            except Exception as e:
                for metric in mets:
                    all_scores[f"{metric.name} [{tag}]"] = [float("nan")] * len(rows)
                all_scores[f"error [{tag}]"] = [str(e)] * len(rows)

        return pd.DataFrame(all_scores)
