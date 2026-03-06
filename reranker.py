# ============================================================
# reranker.py — BM25 keyword-based reranker
#
# Use AFTER vector retrieval to reorder chunks by keyword
# relevance. Best practice: retrieve 2× desired top_k from
# the vector store, then rerank down to final top_k.
#
# BM25 (Best Match 25) is a classic IR ranking function that
# scores documents by term frequency, inverse document
# frequency, and document length normalisation.
# ============================================================

from rank_bm25 import BM25Okapi


class BM25Reranker:
    """
    Reranks a list of retrieved chunks using BM25 keyword scoring.

    Example:
        reranker = BM25Reranker()
        original = rag.retrieve(query, top_k=10)   # fetch 2× candidates
        reranked  = reranker.rerank(query, original, top_k=5)
    """

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Args:
            query:  The user's question string.
            chunks: List of chunk dicts (must have a "text" key).
            top_k:  How many reranked results to return.
                    If None, returns all chunks in reranked order.

        Returns:
            List of chunk dicts in BM25-ranked order.
            Each has two extra keys:
              - bm25_score    : raw BM25 score (higher = more relevant)
              - vector_rank   : original position in the vector results (1-indexed)
        """
        if not chunks:
            return chunks

        tokenized_corpus = [c["text"].lower().split() for c in chunks]
        tokenized_query = query.lower().split()

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        # Zip scores with original index and chunk, sort descending by score
        scored = sorted(
            zip(scores, range(len(chunks)), chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        result = []
        limit = top_k if top_k is not None else len(scored)
        for bm25_score, original_idx, chunk in scored[:limit]:
            c = dict(chunk)                       # shallow copy — don't mutate original
            c["bm25_score"] = round(float(bm25_score), 4)
            c["vector_rank"] = original_idx + 1   # 1-indexed original position
            result.append(c)

        return result
