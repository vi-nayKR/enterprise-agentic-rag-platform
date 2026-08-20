import pytest
from src.rag.models import Document, SearchResult
from src.rag.chunking import SemanticChunker
from src.rag.embeddings import EmbeddingsService
from src.rag.document_store import DocumentStore
from src.rag.rrf import reciprocal_rank_fusion
from src.rag.reranker import CrossEncoderReranker
from src.rag.hybrid_retriever import HybridRetriever


@pytest.mark.asyncio
async def test_semantic_chunker():
    text = "# Main Title\n\nFirst paragraph.\n\n## Sub Title\n\nSecond paragraph."
    doc = Document(filename="test.md", text=text)
    chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    assert all(c.document_id == doc.id for c in chunks)


@pytest.mark.asyncio
async def test_embeddings_service():
    service = EmbeddingsService(dimension=1536)
    vec = await service.embed_query("Test query")
    assert len(vec) == 1536


@pytest.mark.asyncio
async def test_rrf_ranking():
    dense = [{"id": "doc_1", "score": 0.95}, {"id": "doc_2", "score": 0.85}]
    sparse = [
        {"id": "doc_2", "sparse_score": 10.0},
        {"id": "doc_3", "sparse_score": 5.0},
    ]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    assert len(fused) == 3
    # doc_2 appeared in both, so it should rank first with highest RRF score
    assert fused[0].chunk_id == "doc_2"


@pytest.mark.asyncio
async def test_hybrid_retriever_end_to_end():
    retriever = HybridRetriever(top_k=2)
    await retriever.ingest_document(
        filename="rag_intro.md",
        text="LangGraph enables cyclic multi-agent systems with self-reflection.",
    )
    await retriever.ingest_document(
        filename="db_intro.md",
        text="pgvector provides HNSW indexes for cosine distance search.",
    )

    results = await retriever.retrieve("pgvector HNSW indexes")
    assert len(results) > 0
    assert "pgvector" in results[0].text
