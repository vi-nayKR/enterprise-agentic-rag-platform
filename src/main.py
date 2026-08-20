import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from config.settings import settings
from src.api.routes import router as api_router
from src.mcp.manager import mcp_manager
from src.rag.hybrid_retriever import retriever

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize MCP tools and seed baseline documentation
    print(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    await mcp_manager.initialize()
    
    # Ingest baseline platform documentation if empty
    if len(retriever.store.documents) == 0:
        await retriever.ingest_document(
            filename="enterprise_rag_specs.md",
            text="""# Enterprise Agentic RAG Platform Specs
The platform utilizes LangGraph state graphs with supervisor routing, pgvector HNSW cosine indexing,
PostgreSQL BM25 full-text keyword retrieval, Reciprocal Rank Fusion (RRF, k=60), and Anthropic Model Context Protocol (MCP).
Self-reflection graders eliminate hallucinations, ensuring citation-grounded outputs."""
        )
    yield
    print("Shutting down platform resources...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/health", tags=["Health"])
async def root_health():
    return {"status": "healthy", "service": settings.PROJECT_NAME, "version": settings.VERSION}

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui/")

# Mount UI static assets if ui directory exists
if os.path.exists("ui"):
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
