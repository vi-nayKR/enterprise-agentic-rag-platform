from typing import Dict, Any, List
from src.mcp.models import MCPRequest, MCPResponse

class EnterpriseAPIMCPServer:
    """
    Reference MCP Server exposing internal enterprise APIs (Knowledge Tickets, System Health)
    over JSON-RPC 2.0.
    """
    def __init__(self):
        self.tickets = [
            {"id": "TICK-101", "service": "billing", "title": "Stripe webhook retry failure", "status": "resolved", "severity": "medium"},
            {"id": "TICK-102", "service": "auth", "title": "OAuth token refresh intermittent 504", "status": "investigating", "severity": "high"},
            {"id": "TICK-103", "service": "search", "title": "HNSW index build memory spike", "status": "resolved", "severity": "low"},
            {"id": "TICK-104", "service": "database", "title": "Postgres connection pool exhaustion", "status": "open", "severity": "critical"}
        ]
        self.services_health = {
            "api-gateway": {"status": "healthy", "latency_ms": 12.4, "uptime": "99.98%"},
            "vector-search": {"status": "healthy", "latency_ms": 45.2, "uptime": "99.95%"},
            "postgres-db": {"status": "degraded", "latency_ms": 180.5, "uptime": "99.80%"},
            "redis-cache": {"status": "healthy", "latency_ms": 1.2, "uptime": "99.99%"}
        }

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handles incoming JSON-RPC 2.0 requests."""
        if request.method == "tools/list":
            return MCPResponse(
                id=request.id,
                result={
                    "tools": [
                        {
                            "name": "fetch_service_health",
                            "description": "Returns real-time health, latency, and uptime for internal microservices.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "service_name": {
                                        "type": "string",
                                        "description": "Optional name of the specific service (e.g. 'vector-search', 'postgres-db')"
                                    }
                                }
                            }
                        },
                        {
                            "name": "search_knowledge_tickets",
                            "description": "Searches internal engineering incident tickets by service or keyword.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Search keyword or service name"
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    ]
                }
            )

        elif request.method == "tools/call":
            params = request.params or {}
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if tool_name == "fetch_service_health":
                service = arguments.get("service_name")
                if service:
                    data = self.services_health.get(service, {"error": f"Service '{service}' not found."})
                else:
                    data = self.services_health

                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {"type": "text", "text": f"Service Health Report: {data}"}
                        ],
                        "isError": False
                    }
                )

            elif tool_name == "search_knowledge_tickets":
                q = arguments.get("query", "").lower()
                matched = [
                    t for t in self.tickets
                    if q in t["title"].lower() or q in t["service"].lower() or q in t["id"].lower()
                ]

                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {"type": "text", "text": f"Found {len(matched)} tickets: {matched}"}
                        ],
                        "isError": False
                    }
                )

            else:
                return MCPResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Tool '{tool_name}' not found."}
                )

        return MCPResponse(
            id=request.id,
            error={"code": -32600, "message": f"Method '{request.method}' not supported."}
        )
