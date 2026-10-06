import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.rag.chunking import SemanticChunker
from src.rag.models import Document
from src.rag.embeddings import LocalEmbeddings
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import CrossEncoderReranker
from src.rag.models import SearchResult


def test_learned_reranker_constructor_uses_pinned_model():
    factory = Mock()
    with patch.dict("sys.modules", {"sentence_transformers": SimpleNamespace(CrossEncoder=factory)}):
        reranker = CrossEncoderReranker(learned=True)
    factory.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L6-v2",
                                    revision="233902d25c440f23af6f7d6e94d2946bac0bee0a", device="cpu")
    assert reranker.model is factory.return_value


@pytest.mark.asyncio
async def test_all_chunk_modes_preserve_offsets_and_cover_source():
    class Embeddings:
        async def embed_documents(self, texts):
            return [[1.0, 0.0] if index % 2 else [0.0, 1.0] for index in range(len(texts))]

    text = "A long first sentence. A different second topic. A third topic with detail. " * 3
    document = Document(id="source", filename="source.txt", text=text)
    for mode in ("fixed", "recursive", "semantic"):
        chunker = SemanticChunker(chunk_size=60, chunk_overlap=10, mode=mode)
        chunks = await chunker.chunk_document_async(document, Embeddings())
        covered = set()
        for chunk in chunks:
            start, end = chunk.metadata["start_offset"], chunk.metadata["end_offset"]
            assert chunk.text == text[start:end]
            assert len(chunk.text) <= 60
            covered.update(range(start, end))
        assert covered == set(range(len(text)))
    with pytest.raises(ValueError):
        SemanticChunker(chunk_size=10, chunk_overlap=10)


@pytest.mark.asyncio
async def test_local_model_adapters_and_retrieval_mode_switches():
    class Array:
        def tolist(self):
            return [[1.0, 0.0]]

    class Encoder:
        def encode(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            return Array()

        def predict(self, pairs, **kwargs):
            return [-2.0, 4.0]

    embeddings = object.__new__(LocalEmbeddings)
    embeddings.model = Encoder()
    assert await embeddings.embed_query("cat") == [1.0, 0.0]
    assert await embeddings.embed_documents([]) == []
    for mode in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        retriever = HybridRetriever(retrieval_mode=mode, embeddings=embeddings)
        await retriever.ingest_document("cat.txt", "A cat on a mat.")
        assert (await retriever.retrieve("cat", use_cache=False))[0].text == "A cat on a mat."
    reranker = CrossEncoderReranker()
    reranker.model = Encoder()
    results = [SearchResult(chunk_id=name, document_id=name, text=name) for name in ("wrong", "right")]
    assert (await reranker.rerank("query", results))[0].chunk_id == "right"
