import time
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from src.agents.graph import rag_graph
from src.rag.hybrid_retriever import retriever
from src.mcp.manager import mcp_manager
from src.api.sse import stream_rag_events

router = APIRouter()

class QueryRequest(BaseModel):
    query: str = Field(..., description="User question or instruction")
    session_id: Optional[str] = Field(default="default-session", description="Conversation session ID")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)

class Citation(BaseModel):
    citation_id: str
    filename: str
    chunk_id: str
    section: str
    score: float
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    intent: str
    citations: List[Citation]
    thought_trail: List[str]
    latency_ms: float
    is_faithful: bool

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def execute_query(req: QueryRequest):
    """Executes stateful LangGraph multi-agent RAG workflow."""
    start_time = time.time()
    try:
        initial_state = {
            "query": req.query,
            "session_id": req.session_id,
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
        result = await rag_graph.ainvoke(initial_state)
        latency = (time.time() - start_time) * 1000.0

        return QueryResponse(
            answer=result.get("response", "No response generated."),
            intent=result.get("intent", "rag"),
            citations=[Citation(**c) for c in result.get("citations", [])],
            thought_trail=result.get("thought_trail", []),
            latency_ms=round(latency, 2),
            is_faithful=result.get("is_faithful", True)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/stream", tags=["Streaming"])
async def stream_query(req: QueryRequest):
    """Streams live multi-agent reasoning, tool execution, and token generation via SSE."""
    return StreamingResponse(
        stream_rag_events(query=req.query, session_id=req.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.post("/documents/upload", tags=["Ingestion"])
async def upload_document(
    file: UploadFile = File(...),
    category: Optional[str] = Form("general")
):
    """Uploads and ingests a document into the hybrid vector & lexical store."""
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")
        
        doc = await retriever.ingest_document(
            filename=file.filename or "uploaded_doc.txt",
            text=text,
            metadata={"category": category}
        )
        return {
            "status": "success",
            "document_id": doc.id,
            "filename": doc.filename,
            "message": f"Successfully ingested {doc.filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")

@router.get("/documents", tags=["Ingestion"])
async def list_documents():
    """Lists all stored chunks and document repository statistics."""
    chunks = await retriever.store.get_all_chunks()
    return {
        "total_documents": len(retriever.store.documents),
        "total_chunks": len(chunks),
        "documents": [
            {"id": d.id, "filename": d.filename, "created_at": d.created_at}
            for d in retriever.store.documents.values()
        ]
    }

@router.get("/mcp/tools", tags=["MCP"])
async def list_mcp_tools():
    """Returns dynamically discovered tools across all registered MCP servers."""
    tools = await mcp_manager.get_all_tools()
    return {
        "count": len(tools),
        "tools": [{"name": t.name, "description": t.description, "inputSchema": t.inputSchema} for t in tools]
    }

@router.get("/health", tags=["Health"])
async def health_check():
    """Returns status of RAG Subsystems."""
    chunks_count = len(await retriever.store.get_all_chunks())
    tools = await mcp_manager.get_all_tools()
    return {
        "status": "healthy",
        "subsystems": {
            "vector_store": "ready",
            "total_chunks_indexed": chunks_count,
            "mcp_servers": "connected",
            "mcp_tools_count": len(tools),
            "multi_agent_graph": "compiled"
        }
    }
