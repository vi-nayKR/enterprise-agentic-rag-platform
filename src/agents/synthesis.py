from typing import Dict, Any, List
from src.agents.state import AgentState

async def synthesis_node(state: AgentState) -> Dict[str, Any]:
    """
    Citation-Grounded Synthesis Node:
    Synthesizes answers from retrieved documents or MCP tool results,
    attaching explicit hyperlinked citation tags.
    """
    intent = state.get("intent", "rag")
    query = state.get("query", "")
    trail = state.get("thought_trail", [])
    
    citations: List[Dict[str, Any]] = []
    response_text = ""

    if intent == "mcp_tool":
        tool_results = state.get("tool_results", [])
        outputs = "\n".join([f"- **{t['tool']}**: {t['output']}" for t in tool_results])
        response_text = (
            f"### Tool Execution Response\n\n"
            f"Executed tools for query: *'{query}'*\n\n"
            f"{outputs}\n\n"
            f"> *Data retrieved dynamically via Model Context Protocol (MCP).*"
        )

    elif intent == "direct_synthesis":
        response_text = (
            f"Hello! I am your **Enterprise Agentic RAG Platform** assistant.\n\n"
            f"Here is what I can do for you:\n\n"
            f"1. **Hybrid Document Search (RAG)**: Ingest and search technical documentation, PDFs, and policies using **pgvector HNSW cosine indexing** + **PostgreSQL BM25 full-text search** fused with **Reciprocal Rank Fusion (RRF, $k=60$)**.\n"
            f"2. **Database & API Tool Calling (MCP)**: Execute read-only SQL queries against live database schemas and inspect microservice health/tickets over Anthropic **Model Context Protocol (MCP)**.\n"
            f"3. **Self-Reflection & Zero Hallucination**: Automatically evaluate retrieval relevance and synthesize answers backed by verifiable source citations.\n\n"
            f"How can I help you today? You can ask a technical question, drag-and-drop a document, or query database accounts."
        )

    else:
        # RAG Synthesis
        docs = state.get("retrieved_docs", [])
        if not docs:
            response_text = f"I could not find sufficient documentation to answer: *'{query}'*."
        else:
            context_blocks = []
            for i, d in enumerate(docs[:3], 1):
                citation_tag = f"[{d['filename']}#chunk_{d['chunk_id'][:8]}]"
                citations.append({
                    "citation_id": citation_tag,
                    "filename": d["filename"],
                    "chunk_id": d["chunk_id"],
                    "section": d.get("section", "General"),
                    "score": d.get("score", 0.0),
                    "snippet": d["text"][:200]
                })
                context_blocks.append(f"{d['text']} {citation_tag}")

            context_str = "\n\n".join(context_blocks)
            response_text = (
                f"### Grounded Answer\n\n"
                f"Based on the retrieved enterprise documentation:\n\n"
                f"{context_str}\n\n"
                f"*(Synthesized with verified citations for: '{query}')*"
            )

    thought = f"SynthesisAgent: Formatted grounded response with {len(citations)} source citations."
    return {
        "response": response_text,
        "citations": citations,
        "thought_trail": trail + [thought]
    }
