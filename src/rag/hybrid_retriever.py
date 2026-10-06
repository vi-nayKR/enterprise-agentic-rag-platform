import asyncio
from typing import List, Dict, Any, Optional
from src.rag.models import Document, DocumentChunk, SearchResult
from src.rag.chunking import SemanticChunker
from src.rag.embeddings import EmbeddingsService
from src.rag.document_store import DocumentStore
from src.rag.rrf import reciprocal_rank_fusion
from src.rag.reranker import CrossEncoderReranker
from src.rag.cache import SemanticQueryCache
from src.rag.compressor import compressor
from config.settings import settings


class HybridRetriever:
    """
    Shared source-aware retrieval over an in-memory store.
    Supports dense, BM25, RRF, optional learned reranking, exact-key caching,
    and optional extractive compression. Defaults retain the reference model modes.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        rrf_k: int = settings.RRF_K,
        top_k: int = settings.TOP_K,
        retrieval_mode: str = "hybrid_rerank",
        chunking: str = "recursive",
        embeddings=None,
        store=None,
        learned_reranker: bool = False,
    ):
        if retrieval_mode not in {"dense", "bm25", "hybrid", "hybrid_rerank"}:
            raise ValueError("Unknown retrieval mode")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.retrieval_mode = retrieval_mode
        self.chunker = SemanticChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, mode=chunking
        )
        self.embeddings = embeddings if embeddings is not None else EmbeddingsService()
        self.store = store if store is not None else DocumentStore()
        self.reranker = CrossEncoderReranker(top_n=top_k, learned=learned_reranker)
        self.rrf_k = rrf_k
        self.top_k = top_k
        self.cache = SemanticQueryCache()
        self.compressor = compressor

    async def ingest_document(
        self, filename: str, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Ingests, chunks, embeds, and indexes a document."""
        doc = Document(filename=filename, text=text, metadata=metadata or {})
        chunks = await self.chunker.chunk_document_async(doc, self.embeddings)
        if chunks:
            texts = [c.text for c in chunks]
            vectors = await self.embeddings.embed_documents(texts)
            for chunk, vec in zip(chunks, vectors):
                chunk.embedding = vec
        await self.store.add_document(doc)
        await self.store.add_chunks(chunks)

        # Clear cache when new documents are ingested
        self.cache.clear()
        return doc

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        use_compression: bool = True
    ) -> List[SearchResult]:
        """
        Executes cached, parallel dense + sparse retrieval, fuses with RRF,
        reranks, and compresses candidate contexts.
        """
        k = self.top_k if top_k is None else top_k
        if k < 1:
            raise ValueError("top_k must be positive")
        cache_key = f"{query}|top_k={k}|compression={use_compression}"

        # 1. Check Semantic Query Cache
        if use_cache and not filters:
            cached_res = self.cache.get_results(cache_key)
            if cached_res is not None:
                return cached_res[:k]

        # 2. Get Embedding Vector (Cached or Computed)
        dense_results, sparse_results = [], []
        if self.retrieval_mode != "bm25":
            query_vector = self.cache.get_embedding(query) if use_cache else None
            if query_vector is None:
                query_vector = await self.embeddings.embed_query(query)
                if use_cache:
                    self.cache.set_embedding(query, query_vector)
            if self.retrieval_mode == "dense":
                dense_results = await self.store.search_dense(query_vector, top_k=k, filters=filters)
            else:
                dense_results, sparse_results = await asyncio.gather(
                    self.store.search_dense(query_vector, top_k=k * 2, filters=filters),
                    self.store.search_sparse(query, top_k=k * 2, filters=filters),
                )
        else:
            sparse_results = await self.store.search_sparse(query, top_k=k, filters=filters)

        if self.retrieval_mode == "dense":
            final_ranked = dense_results
        elif self.retrieval_mode == "bm25":
            final_ranked = sparse_results
        else:
            fused = reciprocal_rank_fusion(dense_results, sparse_results, k=self.rrf_k)
            final_ranked = await self.reranker.rerank(query, fused, top_n=k) if self.retrieval_mode == "hybrid_rerank" else fused[:k]

        # 6. Extractive Context Compression
        if use_compression:
            final_ranked = self.compressor.compress_results(query, final_ranked)

        results = final_ranked[:k]

        # 7. Store in Cache
        if use_cache and not filters:
            self.cache.set_results(cache_key, results)

        return results


# Global singleton instance for the application
retriever = HybridRetriever()

