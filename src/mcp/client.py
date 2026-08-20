import asyncio
from typing import List, Dict, Any, Optional
from src.mcp.models import MCPRequest, MCPResponse, ToolDefinition, ToolCallResult

class MCPClient:
    """
    Async Client implementing Anthropic MCP JSON-RPC 2.0 protocol.
    Communicates with local/remote MCP servers to discover and invoke tools.
    """
    def __init__(self, server_name: str, server_instance=None):
        self.server_name = server_name
        self.server = server_instance
        self.tools: Dict[str, ToolDefinition] = {}

    async def list_tools(self) -> List[ToolDefinition]:
        """Queries server for registered tools via 'tools/list'."""
        if not self.server:
            return []

        req = MCPRequest(method="tools/list")
        res: MCPResponse = await self.server.handle_request(req)

        if res.error:
            raise RuntimeError(f"Error listing tools from {self.server_name}: {res.error}")

        tool_list = []
        for t in res.result.get("tools", []):
            tool_def = ToolDefinition(**t)
            self.tools[tool_def.name] = tool_def
            tool_list.append(tool_def)
        return tool_list

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> ToolCallResult:
        """Executes a tool on the server via 'tools/call'."""
        if not self.server:
            return ToolCallResult(
                content=[{"type": "text", "text": f"Server '{self.server_name}' not available"}],
                is_error=True
            )

        req = MCPRequest(method="tools/call", params={"name": name, "arguments": arguments})
        res: MCPResponse = await self.server.handle_request(req)

        if res.error:
            return ToolCallResult(
                content=[{"type": "text", "text": f"Tool error: {res.error.get('message', 'Unknown error')}"}],
                is_error=True
            )

        return ToolCallResult(
            content=res.result.get("content", []),
            is_error=res.result.get("isError", False)
        )
