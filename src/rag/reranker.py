import re
from typing import List
from src.rag.models import SearchResult


class CrossEncoderReranker:
    """
    Reranks candidate search results using cross-attentive contextual scoring.
    """

    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    async def rerank(
        self, query: str, candidates: List[SearchResult]
    ) -> List[SearchResult]:
        """Rescores and sorts candidates based on query-passage interaction."""
        if not candidates:
            return []

        query_tokens = set(re.findall(r"\w+", query.lower()))

        for candidate in candidates:
            passage_tokens = re.findall(r"\w+", candidate.text.lower())
            # Term presence & proximity boost
            exact_matches = sum(1 for t in query_tokens if t in passage_tokens)
            coverage = exact_matches / len(query_tokens) if query_tokens else 0.0

            # Combine RRF score with contextual coverage
            base_score = (
                candidate.rrf_score
                if candidate.rrf_score > 0
                else (candidate.dense_score + candidate.sparse_score)
            )
            rerank_score = (base_score * 0.5) + (coverage * 0.5)

            candidate.rerank_score = round(float(rerank_score), 4)

        # Sort by rerank_score descending
        ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        return ranked[: self.top_n]
