from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    Shared execution state across the LangGraph multi-agent network.
    """
    query: str
    session_id: str
    intent: Optional[str]
    retrieved_docs: List[Dict[str, Any]]
    mcp_tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    response: str
    citations: List[Dict[str, Any]]
    is_faithful: bool
    relevance_score: float
    iterations: int
    max_iterations: int
    rewritten_queries: List[str]
    thought_trail: List[str]
