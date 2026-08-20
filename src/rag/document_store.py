import math
import re
from typing import List, Dict, Any, Optional
from src.rag.models import Document, DocumentChunk, SearchResult


class DocumentStore:
    """
    Unified storage engine supporting dense vector search and sparse lexical search.
    Provides an async in-memory & SQLite implementation with pgvector compatibility.
    """

    def __init__(self):
        # In-memory chunk repository: chunk_id -> DocumentChunk
        self.chunks: Dict[str, DocumentChunk] = {}
        # In-memory document repository: doc_id -> Document
        self.documents: Dict[str, Document] = {}

    async def add_document(self, document: Document):
        """Stores a raw document."""
        self.documents[document.id] = document

    async def add_chunks(self, chunks: List[DocumentChunk]):
        """Stores a list of indexed document chunks."""
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

    async def get_all_chunks(self) -> List[DocumentChunk]:
        """Returns all chunks in the store."""
        return list(self.chunks.values())

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculates cosine similarity between two unit-length vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    async def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Performs dense vector similarity search (pgvector HNSW equivalent).
        """
        results: List[SearchResult] = []
        for chunk in self.chunks.values():
            if not chunk.embedding:
                continue

            # Apply metadata filters if provided
            if filters:
                match = all(chunk.metadata.get(k) == v for k, v in filters.items())
                if not match:
                    continue

            score = self._cosine_similarity(query_vector, chunk.embedding)
            results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    dense_score=float(score),
                    metadata=chunk.metadata,
                )
            )

        # Sort descending by dense score
        results.sort(key=lambda r: r.dense_score, reverse=True)
        return results[:top_k]

    async def search_sparse(
        self, query_text: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Performs BM25 / lexical term-matching search (PostgreSQL tsvector equivalent).
        """
        query_terms = set(re.findall(r"\w+", query_text.lower()))
        if not query_terms:
            return []

        total_chunks = max(1, len(self.chunks))
        doc_freq: Dict[str, int] = {}
        for term in query_terms:
            doc_freq[term] = sum(1 for c in self.chunks.values() if term in c.text.lower())

        results: List[SearchResult] = []
        for chunk in self.chunks.values():
            if filters:
                match = all(chunk.metadata.get(k) == v for k, v in filters.items())
                if not match:
                    continue

            chunk_terms = re.findall(r"\w+", chunk.text.lower())
            if not chunk_terms:
                continue

            score = 0.0
            unique_matches = 0
            for term in query_terms:
                tf = chunk_terms.count(term)
                if tf > 0:
                    unique_matches += 1
                    df = doc_freq.get(term, 1)
                    idf = math.log(1.0 + (total_chunks - df + 0.5) / (df + 0.5))
                    norm_len = len(chunk_terms) / 50.0
                    tf_saturated = (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * norm_len))
                    score += idf * tf_saturated

            if unique_matches > 0 and score > 0:
                results.append(
                    SearchResult(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        text=chunk.text,
                        sparse_score=float(round(score, 4)),
                        metadata=chunk.metadata,
                    )
                )

        results.sort(key=lambda r: r.sparse_score, reverse=True)
        return results[:top_k]
