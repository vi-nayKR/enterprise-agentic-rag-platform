import pytest
from src.agents.graph import rag_graph
from src.rag.hybrid_retriever import retriever

@pytest.fixture(autouse=True)
async def setup_test_docs():
    """Ingests baseline documents for multi-agent testing."""
    await retriever.ingest_document(
        filename="system_architecture.md",
        text="LangGraph multi-agent systems use cyclic state graphs for supervisor routing, self-reflection, and tool calling."
    )
    await retriever.ingest_document(
        filename="retrieval_specs.md",
        text="pgvector HNSW index provides sub-10ms cosine nearest neighbor search combined with PostgreSQL BM25."
    )

@pytest.mark.asyncio
async def test_supervisor_routing_and_rag():
    state = {
        "query": "How does LangGraph multi-agent architecture work?",
        "session_id": "test-session-1",
        "intent": None,
        "retrieved_docs": [],
        "mcp_tool_calls": [],
        "tool_results": [],
        "response": "",
        "citations": [],
        "is_faithful": True,
        "relevance_score": 0.0,
        "iterations": 0,
        "max_iterations": 3,
        "rewritten_queries": [],
        "thought_trail": []
    }
    result = await rag_graph.ainvoke(state)
    
    assert result["intent"] == "rag"
    assert len(result["retrieved_docs"]) > 0
    assert len(result["citations"]) > 0
    assert len(result["thought_trail"]) >= 3
    assert "LangGraph" in result["response"]

@pytest.mark.asyncio
async def test_mcp_tool_execution_flow():
    state = {
        "query": "List all customer accounts in the database with Enterprise tier",
        "session_id": "test-session-2",
        "intent": None,
        "retrieved_docs": [],
        "mcp_tool_calls": [],
        "tool_results": [],
        "response": "",
        "citations": [],
        "is_faithful": True,
        "relevance_score": 0.0,
        "iterations": 0,
        "max_iterations": 3,
        "rewritten_queries": [],
        "thought_trail": []
    }
    result = await rag_graph.ainvoke(state)
    
    assert result["intent"] == "mcp_tool"
    assert len(result["tool_results"]) > 0
    assert not result["tool_results"][0]["is_error"]
    assert "Acme Global" in result["response"] or "customer_accounts" in result["response"]

@pytest.mark.asyncio
async def test_direct_synthesis_flow():
    state = {
        "query": "Hello, who are you?",
        "session_id": "test-session-3",
        "intent": None,
        "retrieved_docs": [],
        "mcp_tool_calls": [],
        "tool_results": [],
        "response": "",
        "citations": [],
        "is_faithful": True,
        "relevance_score": 0.0,
        "iterations": 0,
        "max_iterations": 3,
        "rewritten_queries": [],
        "thought_trail": []
    }
    result = await rag_graph.ainvoke(state)
    
    assert result["intent"] == "direct_synthesis"
    assert "Enterprise Agentic RAG Platform" in result["response"]
