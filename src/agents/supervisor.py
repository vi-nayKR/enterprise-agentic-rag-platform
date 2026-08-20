import re
from typing import Dict, Any
from src.agents.state import AgentState

async def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor Agent / Query Router:
    Classifies user query intent into:
    - 'mcp_tool': queries asking for database, tables, SQL, service status, metrics, or tickets
    - 'rag': queries asking about documents, architecture, concepts, policies, or knowledge
    - 'direct_synthesis': general greetings or simple arithmetic
    """
    query = state.get("query", "").lower()
    trail = state.get("thought_trail", [])

    # 1. Conversational & Identity intent (checked with strict word boundaries)
    conversational_regex = r"\b(hi|hello|hey|who are you|what can you do|how can you help|introduce yourself|tell me about yourself|what are your capabilities)\b"
    if re.search(conversational_regex, query) or "help me understand what you can do" in query:
        return {
            "intent": "direct_synthesis",
            "thought_trail": trail + ["Supervisor: Identified conversational greeting / capability query. Routing to SynthesisAgent."]
        }

    # 2. Database & MCP Tool intent
    mcp_keywords = ["table", "database", "sql", "select", "account", "ticket", "health", "uptime", "service", "status", "audit", "latency"]
    if any(re.search(rf"\b{kw}\b", query) for kw in mcp_keywords):
        return {
            "intent": "mcp_tool",
            "thought_trail": trail + [f"Supervisor: Identified database/tool intent. Routing query '{query}' to MCPAgent."]
        }

    # 3. Document / RAG intent (default for domain questions)
    return {
        "intent": "rag",
        "thought_trail": trail + [f"Supervisor: Identified domain document inquiry. Routing query '{query}' to RAGSpecialistAgent."]
    }
