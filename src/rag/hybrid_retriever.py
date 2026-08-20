import asyncio
from typing import List, Dict, Any, Optional
from src.rag.models import Document, DocumentChunk, SearchResult
from src.rag.chunking import SemanticChunker
from src.rag.embeddings import EmbeddingsService
from src.rag.document_store import DocumentStore
from src.rag.rrf import reciprocal_rank_fusion
from src.rag.reranker import CrossEncoderReranker
from src.rag.cache import query_cache
from src.rag.compressor import compressor
from config.settings import settings


class HybridRetriever:
    """
    End-to-End Orchestrator for High-Efficiency Hybrid RAG:
    1. Ingestion: Document -> Semantic Chunks -> Dense Embeddings -> Vector & Lexical Store.
    2. Semantic Cache: Instant sub-10ms lookup for repeated queries.
    3. Multi-Stage Search: Parallel (Dense HNSW + Sparse BM25) -> RRF (k=60) -> Cross-Encoder Rerank.
    4. Extractive Compression: Sentence-level relevance filtering to reduce token overhead.
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        rrf_k: int = settings.RRF_K,
        top_k: int = settings.TOP_K,
    ):
        self.chunker = SemanticChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.embeddings = EmbeddingsService()
        self.store = DocumentStore()
        self.reranker = CrossEncoderReranker(top_n=top_k)
        self.rrf_k = rrf_k
        self.top_k = top_k
        self.cache = query_cache
        self.compressor = compressor

    async def ingest_document(
        self, filename: str, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Ingests, chunks, embeds, and indexes a document."""
        doc = Document(filename=filename, text=text, metadata=metadata or {})
        await self.store.add_document(doc)

        chunks = self.chunker.chunk_document(doc)
        if chunks:
            texts = [c.text for c in chunks]
            vectors = await self.embeddings.embed_documents(texts)
            for chunk, vec in zip(chunks, vectors):
                chunk.embedding = vec
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
        k = top_k or self.top_k

        # 1. Check Semantic Query Cache
        if use_cache and not filters:
            cached_res = self.cache.get_results(query)
            if cached_res is not None:
                return cached_res[:k]

        # 2. Get Embedding Vector (Cached or Computed)
        query_vector = self.cache.get_embedding(query)
        if query_vector is None:
            query_vector = await self.embeddings.embed_query(query)
            self.cache.set_embedding(query, query_vector)

        # 3. Parallel Retrieval Execution
        dense_task = self.store.search_dense(query_vector, top_k=k * 2, filters=filters)
        sparse_task = self.store.search_sparse(query, top_k=k * 2, filters=filters)
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        # 4. Reciprocal Rank Fusion (k=60)
        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=self.rrf_k)

        # 5. Cross-Encoder Rerank
        final_ranked = await self.reranker.rerank(query, fused)

        # 6. Extractive Context Compression
        if use_compression:
            final_ranked = self.compressor.compress_results(query, final_ranked)

        results = final_ranked[:k]

        # 7. Store in Cache
        if use_cache and not filters:
            self.cache.set_results(query, results)

        return results


# Global singleton instance for the application
retriever = HybridRetriever()

