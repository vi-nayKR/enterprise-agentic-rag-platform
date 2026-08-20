import pytest
from src.mcp.models import MCPRequest, MCPResponse, ToolDefinition, ToolCallResult
from src.mcp.client import MCPClient
from src.mcp.servers.database_server import DatabaseMCPServer
from src.mcp.servers.api_server import EnterpriseAPIMCPServer
from src.mcp.manager import MCPManager

@pytest.mark.asyncio
async def test_database_mcp_server_tools_list():
    server = DatabaseMCPServer()
    client = MCPClient(server_name="database", server_instance=server)
    tools = await client.list_tools()
    
    tool_names = [t.name for t in tools]
    assert "list_tables" in tool_names
    assert "query_database" in tool_names

@pytest.mark.asyncio
async def test_database_mcp_server_query_success():
    server = DatabaseMCPServer()
    client = MCPClient(server_name="database", server_instance=server)
    
    result = await client.call_tool("query_database", {"sql_query": "SELECT * FROM customer_accounts WHERE tier = 'Enterprise'"})
    assert not result.is_error
    assert len(result.content) > 0
    assert "Acme Global" in result.content[0]["text"]

@pytest.mark.asyncio
async def test_database_mcp_server_security_guard():
    server = DatabaseMCPServer()
    client = MCPClient(server_name="database", server_instance=server)
    
    # Attempt destructive SQL query
    result = await client.call_tool("query_database", {"sql_query": "DROP TABLE customer_accounts"})
    assert result.is_error
    assert "Security Error" in result.content[0]["text"]

@pytest.mark.asyncio
async def test_api_mcp_server_tools():
    server = EnterpriseAPIMCPServer()
    client = MCPClient(server_name="api", server_instance=server)
    
    # Test service health
    health_res = await client.call_tool("fetch_service_health", {"service_name": "vector-search"})
    assert not health_res.is_error
    assert "healthy" in health_res.content[0]["text"]

    # Test ticket search
    ticket_res = await client.call_tool("search_knowledge_tickets", {"query": "HNSW"})
    assert not ticket_res.is_error
    assert "TICK-103" in ticket_res.content[0]["text"]

@pytest.mark.asyncio
async def test_mcp_manager_aggregation():
    manager = MCPManager()
    tools = await manager.initialize()
    
    tool_names = [t.name for t in tools]
    assert "list_tables" in tool_names
    assert "fetch_service_health" in tool_names
    
    # Invoke routed tool through manager
    res = await manager.call_tool("fetch_service_health", {"service_name": "redis-cache"})
    assert not res.is_error
    assert "99.99%" in res.content[0]["text"]
