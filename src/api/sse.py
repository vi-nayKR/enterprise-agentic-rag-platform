import json
import asyncio
from typing import AsyncGenerator, Dict, Any
from src.agents.graph import rag_graph

async def format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """Formats payload into SSE wire format."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

async def stream_rag_events(query: str, session_id: str = "default") -> AsyncGenerator[str, None]:
    """
    Executes LangGraph agent state graph and yields real-time SSE events:
    - thought: step-by-step reasoning trail
    - tool_call: MCP tool execution details
    - token: synthesized token stream
    - citation: source citations
    - done: completion metrics
    """
    initial_state = {
        "query": query,
        "session_id": session_id,
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

    yield await format_sse_event("status", {"message": f"Processing query: '{query}'"})

    try:
        # Run graph
        result = await rag_graph.ainvoke(initial_state)

        # Stream thoughts
        for thought in result.get("thought_trail", []):
            yield await format_sse_event("thought", {"thought": thought})
            await asyncio.sleep(0.02)

        # Stream tool calls if any
        for tool in result.get("mcp_tool_calls", []):
            yield await format_sse_event("tool_call", tool)

        # Stream response tokens
        response_text = result.get("response", "")
        words = response_text.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield await format_sse_event("token", {"delta": chunk})
            await asyncio.sleep(0.01)

        # Stream citations
        for citation in result.get("citations", []):
            yield await format_sse_event("citation", citation)

        # Done event
        yield await format_sse_event("done", {
            "session_id": session_id,
            "citations_count": len(result.get("citations", [])),
            "intent": result.get("intent")
        })

    except Exception as e:
        yield await format_sse_event("error", {"message": str(e)})
