import pytest
import time
from src.rag.compressor import ContextCompressor
from src.rag.cache import SemanticQueryCache
from src.rag.hybrid_retriever import HybridRetriever

@pytest.mark.asyncio
async def test_context_compressor():
    compressor = ContextCompressor(min_sentence_score=0.20, max_sentences_per_chunk=2)
    query = "pgvector HNSW indexing"
    raw_text = """
    # Vector Search Overview
    PostgreSQL supports relational queries for general enterprise data and relational operations.
    The pgvector extension uses HNSW indexing for sub-10ms dense vector searches across high-dimensional embeddings.
    Weather forecasts in London predict mild rainfall this weekend across various suburbs and countryside trails.
    Additional unrelated background facts regarding office furniture ergonomics and desk lighting fixtures.
    HNSW graphs provide fast approximate nearest neighbor similarity matching for enterprise retrieval tasks.
    """
    compressed = compressor.compress_chunk_text(query, raw_text)
    
    # Assert compression reduced size and kept relevant sentences
    assert len(compressed) < len(raw_text)
    assert "HNSW indexing" in compressed or "pgvector" in compressed
    assert "rainfall" not in compressed

@pytest.mark.asyncio
async def test_semantic_query_cache():
    cache = SemanticQueryCache(max_size=10, default_ttl=60)
    cache.clear()
    
    assert cache.get_results("test query") is None
    assert cache.misses == 1
    
    # Insert mock results
    cache.set_results("test query", [])
    hit_res = cache.get_results("test query")
    assert hit_res is not None
    assert cache.hits == 1

@pytest.mark.asyncio
async def test_retriever_efficiency_pipeline():
    retriever = HybridRetriever()
    await retriever.ingest_document(
        filename="efficiency_test.md",
        text="""# Performance Optimization
        Reciprocal Rank Fusion merges dense vector search with sparse BM25 scores.
        Unrelated filler text about coffee beans and office supplies.
        Context compression strips filler sentences and saves tokens."""
    )
    
    # First query -> Cache Miss
    t0 = time.perf_counter()
    res1 = await retriever.retrieve("Reciprocal Rank Fusion BM25", use_cache=True, use_compression=True)
    t_miss = time.perf_counter() - t0
    
    assert len(res1) > 0
    assert res1[0].metadata.get("compressed_length", 0) <= res1[0].metadata.get("original_length", 1000)
    
    # Second identical query -> Cache Hit (< 5ms)
    t1 = time.perf_counter()
    res2 = await retriever.retrieve("Reciprocal Rank Fusion BM25", use_cache=True)
    t_hit = time.perf_counter() - t1
    
    assert len(res2) == len(res1)
    assert t_hit < t_miss or t_hit < 0.05
