import math

import pytest

from src.rag.document_store import DocumentStore
from src.rag.embeddings import EmbeddingsService
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.models import DocumentChunk


@pytest.mark.asyncio
async def test_retrievers_do_not_share_cached_documents():
    first, second = HybridRetriever(), HybridRetriever()
    await first.ingest_document("first.txt", "The cat sits on a mat.")
    assert await first.retrieve("cat")
    assert await second.retrieve("cat") == []


@pytest.mark.asyncio
async def test_requested_top_k_survives_reranking_and_cache():
    retriever = HybridRetriever(top_k=1)
    for name in ("cat", "dog", "bird"):
        await retriever.ingest_document(name + ".txt", f"{name} is an animal.")
    assert len(await retriever.retrieve("animal", top_k=1)) == 1
    assert len(await retriever.retrieve("animal", top_k=3)) == 3


@pytest.mark.asyncio
async def test_bm25_uses_token_document_frequency_and_actual_mean_length():
    store = DocumentStore()
    await store.add_chunks([
        DocumentChunk(id="cat", document_id="a", chunk_index=0, text="cat"),
        DocumentChunk(id="substring", document_id="b", chunk_index=0, text="concatenate"),
    ])
    results = await store.search_sparse("cat")
    assert [row.chunk_id for row in results] == ["cat"]
    assert results[0].sparse_score == pytest.approx(math.log(2), abs=0.0001)


@pytest.mark.asyncio
async def test_embedding_provider_errors_do_not_change_vector_space():
    class BrokenClient:
        async def aembed_query(self, text):
            raise RuntimeError("provider unavailable")

    service = EmbeddingsService()
    service.use_openai = True
    service._client = BrokenClient()
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await service.embed_query("cat")
