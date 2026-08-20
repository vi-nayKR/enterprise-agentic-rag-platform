from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.supervisor import supervisor_node
from src.agents.rag_agent import rag_agent_node
from src.agents.mcp_agent import mcp_agent_node
from src.agents.reflection import reflection_grader_node
from src.agents.rewriter import query_rewriter_node
from src.agents.synthesis import synthesis_node

# Conditional routing functions
def route_intent_edge(state: AgentState) -> str:
    """Routes based on supervisor intent."""
    intent = state.get("intent", "rag")
    if intent == "mcp_tool":
        return "mcp_agent"
    elif intent == "direct_synthesis":
        return "synthesis"
    return "rag_agent"

def grade_retrieval_edge(state: AgentState) -> str:
    """Evaluates self-reflection grader output to decide if rewrite loop is needed."""
    is_faithful = state.get("is_faithful", True)
    iterations = state.get("iterations", 0)
    max_iterations = state.get("max_iterations", 3)

    if is_faithful or iterations >= max_iterations:
        return "synthesis"
    return "query_rewriter"

# Build StateGraph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("rag_agent", rag_agent_node)
workflow.add_node("mcp_agent", mcp_agent_node)
workflow.add_node("reflection_grader", reflection_grader_node)
workflow.add_node("query_rewriter", query_rewriter_node)
workflow.add_node("synthesis", synthesis_node)

# Set Entry Point
workflow.set_entry_point("supervisor")

# Supervisor Conditional Routing
workflow.add_conditional_edges(
    "supervisor",
    route_intent_edge,
    {
        "rag_agent": "rag_agent",
        "mcp_agent": "mcp_agent",
        "synthesis": "synthesis"
    }
)

# RAG Flow -> Reflection Grader
workflow.add_edge("rag_agent", "reflection_grader")

# Reflection Grader -> (Synthesis OR Query Rewriter)
workflow.add_conditional_edges(
    "reflection_grader",
    grade_retrieval_edge,
    {
        "synthesis": "synthesis",
        "query_rewriter": "query_rewriter"
    }
)

# Query Rewriter Loops Back to RAG Agent!
workflow.add_edge("query_rewriter", "rag_agent")

# MCP Agent -> Synthesis
workflow.add_edge("mcp_agent", "synthesis")

# Synthesis -> END
workflow.add_edge("synthesis", END)

# Compile Graph
rag_graph = workflow.compile()
