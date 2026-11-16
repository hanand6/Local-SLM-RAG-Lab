# ============================================================
# rag_engine.py — Document ingestion + local RAG pipeline
#
# Flow:
#   Upload → chunk → embed (EmbeddingClient) →
#   store in ChromaDB (in-memory) → retrieve → [BM25 rerank] → generate
#
# Configurable at construction and via reconfigure_embedder():
#   - embedding_model : nomic-embed-text | all-MiniLM-L6-v2
#   - similarity      : cosine | l2
#   - chunk_size / chunk_overlap : updated via set_chunker_params()
# ============================================================

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from embedder import EmbeddingClient
from reranker import BM25Reranker
from benchmarker import Benchmarker, BenchmarkResult
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RETRIEVAL,
    MODELS,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    GEMINI_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    EMBEDDING_MODEL,
    INDEX_BATCH_SIZE,
)


# ── Document chunker (no LangChain dep) ──────────────────────

class TextChunker:
    """
    Recursive character text splitter.
    Mirrors LangChain's RecursiveCharacterTextSplitter logic
    without the dependency overhead.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        chunks = self._split_recursive(text, self.SEPARATORS)
        return [c.strip() for c in chunks if c.strip()]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        sep = ""
        remaining = separators.copy()
        while remaining:
            sep = remaining.pop(0)
            if sep == "" or sep in text:
                break

        splits = text.split(sep) if sep else list(text)
        chunks, current = [], ""

        for part in splits:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    overlap_text = current[-self.overlap:] if self.overlap else ""
                    current = overlap_text + (sep if overlap_text else "") + part
                else:
                    sub = self._split_recursive(part, remaining or [""])
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""

        if current:
            chunks.append(current)

        return chunks


# ── Document loaders ─────────────────────────────────────────

def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_pdf(path: str) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(path)
        return "\n\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")


def load_docx(path: str) -> str:
    try:
        import docx
        doc = docx.Document(path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")


def load_document(path: str) -> str:
    """Dispatch to correct loader based on file extension."""
    ext = Path(path).suffix.lower()
    loaders = {
        ".pdf":  load_pdf,
        ".docx": load_docx,
        ".doc":  load_docx,
        ".txt":  load_text_file,
        ".md":   load_text_file,
        ".csv":  load_text_file,
    }
    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader(path)


# ── RAG Engine ───────────────────────────────────────────────

class RAGEngine:
    """
    Local RAG pipeline with configurable embedding model,
    similarity metric, chunk settings, and optional BM25 reranking.

    Privacy: embeddings (SentenceTransformers by default) and the vector
             store always run locally. Ollama models stay 100% local;
             cloud models (OpenRouter etc.) receive the retrieved chunks.
    """

    def __init__(
        self,
        collection_name: str = "rag_store",
        embedding_model: str = EMBEDDING_MODEL,
        similarity: str = "cosine",         # "cosine" | "l2"
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        persist_dir: str | None = None,
    ):
        """
        persist_dir: if set, the index is stored on disk there and survives
        restarts. Each (embedding model, similarity) pair gets its own
        collection, so switching models never mixes vector spaces and
        switching back finds the old index again. If None, the index lives
        in memory only.
        """
        self.embedding_model = embedding_model
        self.similarity = similarity
        self._base_name = collection_name
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._collection_name = self._name_for(embedding_model, similarity)

        self.embed_client = EmbeddingClient(embedding_model)
        self.reranker     = BM25Reranker()
        self.benchmarker  = Benchmarker()
        self.chunker      = TextChunker(chunk_size, chunk_overlap)

        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._chroma = chromadb.PersistentClient(
                path=str(self._persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._chroma = chromadb.Client(Settings(anonymized_telemetry=False))
        self._collection = self._make_collection(self._collection_name, similarity)
        self._indexed_docs: list[dict] = self._load_docs()

    # ── Persistence helpers ──────────────────────────────────

    @property
    def persistent(self) -> bool:
        return self._persist_dir is not None

    def _name_for(self, embedding_model: str, similarity: str) -> str:
        if not getattr(self, "_persist_dir", None):
            return self._base_name
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", embedding_model).strip("-").lower()
        return f"{self._base_name}-{slug}-{similarity}"[:60]

    def _docs_file(self) -> Path | None:
        return self._persist_dir / f"{self._collection_name}.docs.json" if self._persist_dir else None

    def _load_docs(self) -> list[dict]:
        f = self._docs_file()
        if f and f.exists():
            try:
                docs = json.loads(f.read_text(encoding="utf-8"))
                # Drop entries whose chunks are no longer in the store.
                return [d for d in docs if self._collection.get(
                    where={"source": d.get("source", d["doc_name"])}, limit=1)["ids"]]
            except (json.JSONDecodeError, KeyError, ValueError):
                return []
        return []

    def _save_docs(self) -> None:
        f = self._docs_file()
        if f:
            f.write_text(json.dumps(self._indexed_docs, indent=2), encoding="utf-8")

    # ── Configuration ────────────────────────────────────────

    def set_chunker_params(self, chunk_size: int, overlap: int) -> None:
        """Update chunk settings — takes effect on the next index_file call."""
        self.chunker.chunk_size = chunk_size
        self.chunker.overlap    = overlap

    def reconfigure_embedder(
        self,
        embedding_model: str,
        similarity: str,
    ) -> bool:
        """
        Change embedding model or similarity metric.
        In memory: resets the index (different vector space).
        Persistent: switches to that model's own saved index.
        Returns True if a change was made.
        """
        changed = (
            embedding_model != self.embedding_model
            or similarity != self.similarity
        )
        if not changed:
            return False

        self.embedding_model = embedding_model
        self.similarity      = similarity
        self.embed_client    = EmbeddingClient(embedding_model)
        if self.persistent:
            # Switch to (or create) the index for this model + metric.
            self._collection_name = self._name_for(embedding_model, similarity)
            self._collection = self._make_collection(self._collection_name, similarity)
            self._indexed_docs = self._load_docs()
        else:
            self.reset()      # wipe index — incompatible embeddings
        return True

    def _make_collection(self, name: str, similarity: str):
        """Create or get a ChromaDB collection with the given similarity space."""
        return self._chroma.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": similarity},
        )

    def _similarity_from_distance(self, dist: float) -> float:
        """Convert ChromaDB distance score to a [0,1] similarity value."""
        if self.similarity == "cosine":
            return round(1 - dist, 4)          # cosine dist ∈ [0,2]
        else:                                   # l2 (squared euclidean)
            return round(1 / (1 + dist), 4)    # l2 dist ∈ [0, ∞)

    # ── Indexing ─────────────────────────────────────────────

    def index_file(self, file_path: str, progress_cb=None) -> dict:
        """
        Load → chunk → embed → store.
        Uses current chunker settings (chunk_size, overlap).
        Embeds and stores in batches (ChromaDB caps a single insert at
        ~5,461 items, and batching lets the UI show progress).

        progress_cb(done_chunks, total_chunks) is called after each batch.
        Returns indexing stats.
        """
        t0 = time.perf_counter()
        doc_name = Path(file_path).name

        # 1. Load
        raw_text = load_document(file_path)

        # Re-indexing the same file replaces its old chunks.
        self.remove_document(doc_name, save=False)

        # 2. Chunk
        chunks = self.chunker.split(raw_text)
        total = len(chunks)

        # 3 + 4. Embed (local) and store, batch by batch
        for start in range(0, total, INDEX_BATCH_SIZE):
            batch = chunks[start:start + INDEX_BATCH_SIZE]
            idxs = range(start, start + len(batch))
            self._collection.add(
                embeddings=self.embed_client.embed_batch(batch),
                documents=batch,
                ids=[f"{doc_name}::chunk::{i}" for i in idxs],
                metadatas=[{"source": doc_name, "chunk_idx": i} for i in idxs],
            )
            if progress_cb:
                progress_cb(start + len(batch), total)

        elapsed = time.perf_counter() - t0
        stats = {
            "doc_name":      doc_name,
            "char_count":    len(raw_text),
            "chunks":        len(chunks),
            "avg_chunk_len": round(len(raw_text) / len(chunks)) if chunks else 0,
            "index_time_s":  round(elapsed, 2),
            "embedding_model": self.embedding_model,
            "similarity":    self.similarity,
            "source":        doc_name,       # name stored on the chunks
            "indexed_at":    time.strftime("%Y-%m-%d %H:%M"),
        }
        self._indexed_docs.append(stats)
        self._save_docs()
        return stats

    def remove_document(self, source: str, save: bool = True) -> bool:
        """Delete one document's chunks. Returns True if anything was removed."""
        existing = self._collection.get(where={"source": source}, limit=1)["ids"]
        if existing:
            self._collection.delete(where={"source": source})
        before = len(self._indexed_docs)
        self._indexed_docs = [d for d in self._indexed_docs if d.get("source", d["doc_name"]) != source]
        if save:
            self._save_docs()
        return bool(existing) or len(self._indexed_docs) != before

    def index_text(self, text: str, doc_name: str = "pasted_text") -> dict:
        """Index raw text directly (for paste-in content)."""
        import tempfile
        # Name the temp file after doc_name so citations show the right source.
        safe = "".join(c if c.isalnum() or c in "-_ ." else "_" for c in doc_name).strip() or "pasted_text"
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / f"{safe}.txt"
            path.write_text(text, encoding="utf-8")
            result = self.index_file(str(path))
        result["doc_name"] = doc_name      # display name; chunks keep result["source"]
        self._save_docs()
        return result

    # ── Retrieval ────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """
        Embed query → cosine/l2 search → return top_k chunks with scores.
        """
        if self._collection.count() == 0:
            return []

        query_emb = self.embed_client.embed(query)
        results   = self._collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text":       doc,
                "source":     meta.get("source", "unknown"),
                "chunk_idx":  meta.get("chunk_idx", -1),
                "similarity": self._similarity_from_distance(dist),
            })
        return chunks

    def retrieve_reranked(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        fetch_k: int | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Two-stage retrieval: vector search → BM25 rerank.

        Args:
            query:   User question.
            top_k:   Final number of chunks to return.
            fetch_k: Candidate pool for reranker (default: 2 × top_k).

        Returns:
            (vector_chunks, bm25_chunks)
            vector_chunks : top_k chunks in original vector order
            bm25_chunks   : top_k chunks in BM25-reranked order
        """
        if fetch_k is None:
            fetch_k = max(top_k * 2, 10)

        # Stage 1 — wide vector retrieval
        candidates = self.retrieve(query, top_k=fetch_k)
        vector_chunks = candidates[:top_k]

        # Stage 2 — BM25 rerank from the wider pool
        bm25_chunks = self.reranker.rerank(query, candidates, top_k=top_k)

        return vector_chunks, bm25_chunks

    # ── Prompt builder ───────────────────────────────────────

    @staticmethod
    def build_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
        """
        Returns (system_prompt, user_prompt) for RAG generation.
        """
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}, chunk {c['chunk_idx']+1}]\n{c['text']}"
            for c in chunks
        )
        system = (
            "You are a precise, factual document Q&A assistant.\n"
            "Rules:\n"
            "1. Answer ONLY from the provided context sections.\n"
            "2. If the answer is not in the context, respond: "
            "'⚠️ Not found in the indexed documents.'\n"
            "3. Be concise. Cite the source/chunk when quoting.\n"
            "4. Do not hallucinate or add external knowledge."
        )
        user = (
            f"Context sections:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return system, user

    # ── Internal: cloud model routing ────────────────────────

    def _run_cloud_model(
        self,
        model: str,
        system: str,
        prompt: str,
        run_index: int = 0,
        on_token=None,
    ) -> BenchmarkResult:
        """Route cloud models to the appropriate API client.
        on_token(text_so_far) streams the answer (OpenAI-compatible backends)."""
        from api_client import OpenAICompatClient, NvidiaApiClient, GeminiApiClient
        backend = MODELS.get(model, {}).get("backend", "ollama")

        if backend == "openrouter":
            if not OPENROUTER_API_KEY:
                result = BenchmarkResult(model=model, run_index=run_index, prompt_preview=prompt[:120])
                result.error = (
                    "OPENROUTER_API_KEY not set. Get a free key at https://openrouter.ai/keys "
                    "and add it to your .env file."
                )
                return result
            client = OpenAICompatClient(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                model=model,
            )
            return client.generate(prompt, system=system, run_index=run_index, on_token=on_token)

        elif backend == "nvidia_nim":
            if not NVIDIA_API_KEY:
                result = BenchmarkResult(model=model, run_index=run_index, prompt_preview=prompt[:120])
                result.error = "NVIDIA_API_KEY not set. Add it to your .env file."
                return result
            client = NvidiaApiClient(
                api_key=NVIDIA_API_KEY,
                base_url=NVIDIA_BASE_URL,
                model=model,
            )
            return client.generate(prompt, system=system, run_index=run_index, on_token=on_token)

        elif backend == "gemini":
            if not GEMINI_API_KEY:
                result = BenchmarkResult(model=model, run_index=run_index, prompt_preview=prompt[:120])
                result.error = "GEMINI_API_KEY not set. Add it to your .env file."
                return result
            # ⚠️  Use GEMINI_MODEL constant (e.g. 'gemini-2.5-flash'), NOT the
            #     MODELS dict key ('google/gemini-2.5-flash') — the API rejects the prefix.
            from config import GEMINI_MODEL
            client = GeminiApiClient(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)
            return client.generate(prompt, system=system, run_index=run_index, on_token=on_token)

        else:
            result = BenchmarkResult(model=model, run_index=run_index, prompt_preview=prompt[:120])
            result.error = f"Unknown backend '{backend}' for model {model}"
            return result

    def _generate(
        self,
        question: str,
        chunks: list[dict],
        model: str,
        run_index: int = 0,
    ) -> BenchmarkResult:
        """Run LLM generation from a set of chunks. Handles both Ollama and cloud."""
        system, prompt = self.build_prompt(question, chunks)
        backend = MODELS.get(model, {}).get("backend", "ollama")
        if backend == "ollama":
            return self.benchmarker.run_single(model=model, prompt=prompt, system=system, run_index=run_index)
        else:
            return self._run_cloud_model(model=model, system=system, prompt=prompt, run_index=run_index)

    # ── Full RAG query ───────────────────────────────────────

    def query(
        self,
        question: str,
        model: str,
        top_k: int = TOP_K_RETRIEVAL,
    ) -> dict:
        """
        Standard RAG query (vector retrieval only).
        Returns a result dict with answer + performance metrics.
        """
        t_ret = time.perf_counter()
        chunks = self.retrieve(question, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        if not chunks:
            return {
                "model": model, "question": question,
                "answer": "⚠️ No documents indexed yet. Please upload and index a document first.",
                "retrieval_ms": retrieval_ms, "generation_ms": 0,
                "total_ms": retrieval_ms, "tokens_per_sec": 0,
                "ttft_ms": 0, "output_tokens": 0,
                "context_chunks": [], "benchmark_result": None,
            }

        bm_result = self._generate(question, chunks, model)

        # Surface any API / model errors visibly in the answer field
        answer = bm_result.response or (
            f"❌ API error: {bm_result.error}" if bm_result.error else "⚠️ No response returned."
        )

        return {
            "model":             model,
            "question":          question,
            "answer":            answer,
            "retrieval_ms":      round(retrieval_ms, 1),
            "generation_ms":     round(bm_result.eval_duration_ms, 1),
            "total_ms":          round(retrieval_ms + bm_result.total_duration_ms, 1),
            "tokens_per_sec":    round(bm_result.tokens_per_second, 1),
            "ttft_ms":           round(bm_result.ttft_ms, 1),
            "output_tokens":     bm_result.output_tokens,
            "context_chunks":    chunks,
            "benchmark_result":  bm_result,
        }

    def query_compare(
        self,
        question: str,
        model: str,
        top_k: int = TOP_K_RETRIEVAL,
    ) -> tuple[dict, dict]:
        """
        Run the same question with both vector-only and vector+BM25 retrieval.
        Runs the LLM twice (once per chunk ordering).

        Returns:
            (vector_result, bm25_result)
            Both are standard result dicts (same schema as query()).
        """
        # ── Wide retrieval (2× candidates)  ──────────────────
        t_ret = time.perf_counter()
        vector_chunks, bm25_chunks = self.retrieve_reranked(question, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        if not vector_chunks:
            empty = {
                "model": model, "question": question,
                "answer": "⚠️ No documents indexed yet.",
                "retrieval_ms": retrieval_ms, "generation_ms": 0,
                "total_ms": retrieval_ms, "tokens_per_sec": 0,
                "ttft_ms": 0, "output_tokens": 0,
                "context_chunks": [], "benchmark_result": None,
            }
            return empty, empty

        # ── Vector result ─────────────────────────────────────
        bm_vec = self._generate(question, vector_chunks, model, run_index=0)
        vec_answer = bm_vec.response or (
            f"❌ API error: {bm_vec.error}" if bm_vec.error else "⚠️ No response returned."
        )
        vector_result = {
            "model":            model,
            "question":         question,
            "answer":           vec_answer,
            "retrieval_ms":     round(retrieval_ms / 2, 1),
            "generation_ms":    round(bm_vec.eval_duration_ms, 1),
            "total_ms":         round(retrieval_ms / 2 + bm_vec.total_duration_ms, 1),
            "tokens_per_sec":   round(bm_vec.tokens_per_second, 1),
            "ttft_ms":          round(bm_vec.ttft_ms, 1),
            "output_tokens":    bm_vec.output_tokens,
            "context_chunks":   vector_chunks,
            "benchmark_result": bm_vec,
        }

        # ── BM25 reranked result ──────────────────────────────
        bm_bm25 = self._generate(question, bm25_chunks, model, run_index=1)
        bm25_answer = bm_bm25.response or (
            f"❌ API error: {bm_bm25.error}" if bm_bm25.error else "⚠️ No response returned."
        )
        bm25_result = {
            "model":            model,
            "question":         question,
            "answer":           bm25_answer,
            "retrieval_ms":     round(retrieval_ms / 2, 1),
            "generation_ms":    round(bm_bm25.eval_duration_ms, 1),
            "total_ms":         round(retrieval_ms / 2 + bm_bm25.total_duration_ms, 1),
            "tokens_per_sec":   round(bm_bm25.tokens_per_second, 1),
            "ttft_ms":          round(bm_bm25.ttft_ms, 1),
            "output_tokens":    bm_bm25.output_tokens,
            "context_chunks":   bm25_chunks,
            "benchmark_result": bm_bm25,
        }

        return vector_result, bm25_result

    def get_all_text(self, max_chars: int) -> tuple[str, int]:
        """
        Reassemble indexed documents in reading order (for "long-context"
        comparisons). Returns (text, total_chars_available) — text is cut
        at max_chars. Chunk overlap means a little text repeats.
        """
        if self._collection.count() == 0:
            return "", 0
        data = self._collection.get(include=["documents", "metadatas"])
        rows = sorted(
            zip(data["metadatas"], data["documents"]),
            key=lambda r: (r[0].get("source", ""), r[0].get("chunk_idx", 0)),
        )
        parts, current = [], None
        for meta, doc in rows:
            if meta.get("source") != current:
                current = meta.get("source")
                parts.append(f"\n\n===== {current} =====\n")
            parts.append(doc)
        full = "\n".join(parts).strip()
        return full[:max_chars], len(full)

    # ── Stats ────────────────────────────────────────────────

    @property
    def chunk_count(self) -> int:
        return self._collection.count()

    @property
    def indexed_docs(self) -> list[dict]:
        return self._indexed_docs

    # ── Reset ────────────────────────────────────────────────

    def reset(self) -> None:
        """Wipe all indexed chunks. Recreates the collection with current similarity."""
        self._chroma.delete_collection(self._collection_name)
        self._collection = self._make_collection(self._collection_name, self.similarity)
        self._indexed_docs.clear()
        self._save_docs()
