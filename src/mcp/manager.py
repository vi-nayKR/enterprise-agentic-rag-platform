from typing import Dict, List, Any, Optional
from src.mcp.models import ToolDefinition, ToolCallResult
from src.mcp.client import MCPClient
from src.mcp.servers.database_server import DatabaseMCPServer
from src.mcp.servers.api_server import EnterpriseAPIMCPServer

class MCPManager:
    """
    Unified manager pooling multiple MCP servers and routing tool executions.
    """
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
        self.tool_to_server_map: Dict[str, str] = {}
        self.cached_tools: Dict[str, ToolDefinition] = {}
        self._register_default_servers()

    def _register_default_servers(self):
        """Registers the built-in reference MCP servers."""
        self.register_server("database", DatabaseMCPServer())
        self.register_server("enterprise_api", EnterpriseAPIMCPServer())

    def register_server(self, name: str, server_instance):
        """Adds an MCP server instance to the pool."""
        self.clients[name] = MCPClient(server_name=name, server_instance=server_instance)

    async def initialize(self) -> List[ToolDefinition]:
        """Discovers all tools across all connected MCP servers."""
        all_tools: List[ToolDefinition] = []
        for server_name, client in self.clients.items():
            tools = await client.list_tools()
            for t in tools:
                self.tool_to_server_map[t.name] = server_name
                self.cached_tools[t.name] = t
                all_tools.append(t)
        return all_tools

    async def get_all_tools(self) -> List[ToolDefinition]:
        """Returns cached tool definitions or initializes discovery."""
        if not self.cached_tools:
            return await self.initialize()
        return list(self.cached_tools.values())

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolCallResult:
        """Routes a tool call to the appropriate MCP client."""
        if not self.tool_to_server_map:
            await self.initialize()

        server_name = self.tool_to_server_map.get(name)
        if not server_name or server_name not in self.clients:
            return ToolCallResult(
                content=[{"type": "text", "text": f"Error: Tool '{name}' is not registered across any MCP server."}],
                is_error=True
            )

        client = self.clients[server_name]
        return await client.call_tool(name, arguments)

mcp_manager = MCPManager()
