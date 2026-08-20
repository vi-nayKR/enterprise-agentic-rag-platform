import re
from typing import Dict, Any
from src.agents.state import AgentState

async def reflection_grader_node(state: AgentState) -> Dict[str, Any]:
    """
    Self-Reflection & Hallucination Grader:
    Evaluates retrieved context quality, scores relevance,
    and flags whether the answer requires query rewriting.
    """
    query = state.get("rewritten_queries", [])[-1] if state.get("rewritten_queries") else state.get("query", "")
    docs = state.get("retrieved_docs", [])
    trail = state.get("thought_trail", [])
    iterations = state.get("iterations", 0)

    if not docs:
        thought = f"Grader (Iter {iterations + 1}): Zero documents retrieved. Triggering Query Rewriter."
        return {
            "relevance_score": 0.0,
            "is_faithful": False,
            "iterations": iterations + 1,
            "thought_trail": trail + [thought]
        }

    # Evaluate keyword coverage on substantive terms
    stopwords = {"the", "a", "an", "is", "are", "and", "or", "to", "in", "for", "with", "of", "on", "as", "by", "this", "that", "how", "what", "why", "does", "when", "where"}
    query_tokens = set(re.findall(r"\w+", query.lower())) - stopwords
    if not query_tokens:
        query_tokens = set(re.findall(r"\w+", query.lower()))

    combined_doc_text = " ".join([d["text"] for d in docs]).lower()

    matches = 0
    for q in query_tokens:
        if q in combined_doc_text or any(len(q) >= 4 and w.startswith(q[:4]) for w in re.findall(r"\w+", combined_doc_text)):
            matches += 1

    relevance_score = matches / len(query_tokens) if query_tokens else 1.0

    # Faithfulness threshold: 0.60
    is_faithful = relevance_score >= 0.60

    status = "Passed relevance threshold" if is_faithful else "Low relevance / potential hallucination risk"
    thought = f"Grader (Iter {iterations + 1}): Evaluated {len(docs)} documents. Substantive Relevance Score: {relevance_score:.2f} ({status})."

    return {
        "relevance_score": round(relevance_score, 2),
        "is_faithful": is_faithful,
        "iterations": iterations + 1,
        "thought_trail": trail + [thought]
    }
