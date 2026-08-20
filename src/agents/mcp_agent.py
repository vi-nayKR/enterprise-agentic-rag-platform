import re
from typing import Dict, Any
from src.agents.state import AgentState
from src.mcp.manager import mcp_manager

async def mcp_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    MCP Tool Specialist Agent:
    Inspects user query, matches appropriate MCP tool, executes it over JSON-RPC 2.0,
    and captures tool outputs.
    """
    query = state.get("query", "")
    trail = state.get("thought_trail", [])
    q_lower = query.lower()

    tool_results = []
    tool_calls = []

    # Simple intent-to-tool mapper (or dynamic LLM tool caller)
    if "table" in q_lower or "schema" in q_lower:
        tool_name = "list_tables"
        args = {}
    elif "select" in q_lower or "account" in q_lower or "spend" in q_lower or "user" in q_lower:
        tool_name = "query_database"
        # Generate appropriate SELECT SQL
        if "enterprise" in q_lower:
            sql = "SELECT * FROM customer_accounts WHERE tier = 'Enterprise'"
        elif "suspended" in q_lower or "status" in q_lower:
            sql = "SELECT * FROM customer_accounts WHERE status = 'suspended'"
        else:
            sql = "SELECT * FROM customer_accounts LIMIT 5"
        args = {"sql_query": sql}
    elif "ticket" in q_lower or "incident" in q_lower or "jira" in q_lower:
        tool_name = "search_knowledge_tickets"
        # Extract query term
        term = "billing" if "billing" in q_lower else ("auth" if "auth" in q_lower else "search")
        args = {"query": term}
    elif "health" in q_lower or "status" in q_lower or "uptime" in q_lower:
        tool_name = "fetch_service_health"
        svc = "vector-search" if "vector" in q_lower else ("postgres-db" if "postgres" in q_lower else None)
        args = {"service_name": svc} if svc else {}
    else:
        tool_name = "fetch_service_health"
        args = {}

    tool_calls.append({"name": tool_name, "arguments": args})
    res = await mcp_manager.call_tool(tool_name, args)
    tool_results.append({
        "tool": tool_name,
        "is_error": res.is_error,
        "output": res.content[0]["text"] if res.content else "No output"
    })

    thought = f"MCPAgent: Invoked tool '{tool_name}' with args {args}. Result status: {'Error' if res.is_error else 'Success'}."
    return {
        "mcp_tool_calls": tool_calls,
        "tool_results": tool_results,
        "thought_trail": trail + [thought]
    }
