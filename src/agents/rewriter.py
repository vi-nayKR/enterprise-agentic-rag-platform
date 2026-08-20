from typing import Dict, Any
from src.agents.state import AgentState

async def query_rewriter_node(state: AgentState) -> Dict[str, Any]:
    """
    Query Rewriter Node:
    Reformulates and expands search queries when retrieval relevance is low.
    """
    original_query = state.get("query", "")
    rewritten_list = state.get("rewritten_queries", [])
    trail = state.get("thought_trail", [])
    
    iterations = len(rewritten_list)
    
    if iterations == 0:
        # Step 1: Expand with domain concept synonyms
        expansion = "architecture specifications implementation details"
        new_query = f"{original_query} {expansion}".strip()
    elif iterations == 1:
        # Step 2: Extract core substantive keywords
        import re
        stopwords = {"how", "what", "why", "does", "the", "is", "a", "an", "in", "for", "of", "to", "when", "where"}
        tokens = [w for w in re.findall(r"\w+", original_query.lower()) if w not in stopwords]
        new_query = " ".join(tokens)
    else:
        # Step 3: Generalized conceptual fallback
        new_query = f"{original_query} overview specs"

    thought = f"QueryRewriter (Iter {iterations + 1}): Reformulated query to '{new_query}' for improved domain recall."
    return {
        "rewritten_queries": rewritten_list + [new_query],
        "thought_trail": trail + [thought]
    }
