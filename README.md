# Agentic RAG learning reference

A FastAPI and LangGraph prototype for exploring document retrieval, query routing, citations, and tool calls. This is a **learning reference**, not a deployed enterprise service or a measured high-accuracy RAG system.

## What the code implements

- Document ingestion, chunking, and an **in-memory** document store (`src/rag/document_store.py`). Documents and chunks disappear when the process restarts.
- Dense similarity over generated vectors, a Python BM25-like lexical scorer, reciprocal rank fusion, reranking, and context compression. An OpenAI embedding provider can be configured; without it, the code uses deterministic token-hash vectors.
- A LangGraph route through retrieval, heuristic reflection, bounded query rewriting, and synthesis (`src/agents/graph.py`).
- FastAPI routes for uploading text, querying, streaming events, listing documents, and discovering built-in tools (`src/api/routes.py`).
- In-process tool examples for a database-shaped data source and an API-shaped data source. `src/mcp/client.py` calls local Python server objects using JSON-RPC-like request models; this repository has not demonstrated an independently interoperable MCP client/server connection.
- A small evaluation dataset and token-overlap heuristics (`src/evals/metrics.py`). These metrics are **not Ragas metrics** and do not establish factual correctness or hallucination rates.

The `docker-compose.yml` file starts supporting services for experimentation. The inspected retrieval path uses the in-memory store; it does not use pgvector HNSW or PostgreSQL full-text search. Descriptions in older `docs/phase*.md` files are historical design notes and may discuss proposed capabilities beyond the current code.

## Run locally

Python 3.12 is recommended.

```bash
git clone https://github.com/vi-nayKR/enterprise-agentic-rag-platform.git
cd enterprise-agentic-rag-platform
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. Upload a small text document with `POST /documents/upload`, then ask a question with `POST /query`. The API does not have authentication or upload limits; run it locally with non-sensitive sample data.

```bash
python -m pytest tests -q
python -m src.evals.ragas_pipeline
```

The evaluation command's historical name contains `ragas`, but its implementation uses local token-overlap scoring. No versioned, held-out, live-provider result is published here. Numeric recall, latency, and cost improvement claims require a fixed dataset, baseline, environment, and recorded run before they can be made.

## What to improve next

1. Choose one real retrieval task and create a held-out dataset with answer and citation labels.
2. Compare lexical, dense, and fused retrieval on the same cases, reporting failures and latency.
3. If persistence is needed, implement and exercise PostgreSQL/pgvector rather than treating the current in-memory search as equivalent.
4. If cross-client tool interoperability is needed, use a protocol SDK and test a separate MCP client and server with appropriate authorization.
5. Add authentication, tenant boundaries, upload validation, and rate limits before exposing the API to untrusted users.

This repository has no checked-in license file. Contact the author before reusing its code outside the terms GitHub provides for viewing and forking.
