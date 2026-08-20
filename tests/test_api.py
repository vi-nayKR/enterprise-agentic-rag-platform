import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app

@pytest.mark.asyncio
async def test_health_endpoints():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"

        v1_res = await ac.get("/api/v1/health")
        assert v1_res.status_code == 200
        v1_data = v1_res.json()
        assert v1_data["subsystems"]["vector_store"] == "ready"

@pytest.mark.asyncio
async def test_mcp_tools_list_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/mcp/tools")
        assert res.status_code == 200
        data = res.json()
        assert data["count"] >= 4
        tool_names = [t["name"] for t in data["tools"]]
        assert "list_tables" in tool_names
        assert "fetch_service_health" in tool_names

@pytest.mark.asyncio
async def test_document_upload_and_list():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Upload
        files = {"file": ("test_doc.md", b"# Testing Document\nThis is a test content for API ingestion.", "text/markdown")}
        upload_res = await ac.post("/api/v1/documents/upload", files=files, data={"category": "testing"})
        assert upload_res.status_code == 200
        upload_data = upload_res.json()
        assert upload_data["status"] == "success"

        # List
        list_res = await ac.get("/api/v1/documents")
        assert list_res.status_code == 200
        list_data = list_res.json()
        assert list_data["total_documents"] >= 1

@pytest.mark.asyncio
async def test_query_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {"query": "Explain pgvector HNSW indexing", "session_id": "api-test"}
        res = await ac.post("/api/v1/query", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "answer" in data
        assert data["intent"] == "rag"
        assert len(data["thought_trail"]) > 0

@pytest.mark.asyncio
async def test_sse_streaming_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {"query": "List database accounts with Enterprise tier", "session_id": "stream-test"}
        res = await ac.post("/api/v1/query/stream", json=payload)
        assert res.status_code == 200
        text = res.text
        assert "event: status" in text
        assert "event: thought" in text
        assert "event: token" in text
        assert "event: done" in text
