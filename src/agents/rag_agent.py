from typing import Dict, Any
from src.agents.state import AgentState
from src.rag.hybrid_retriever import retriever

async def rag_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    RAG Specialist Agent:
    Executes hybrid retrieval (Dense HNSW + Sparse BM25 + RRF + Reranker)
    with HyDE (Hypothetical Document Embeddings) expansion and context compression.
    """
    # Use rewritten query if present from self-reflection loop
    query = state.get("rewritten_queries", [])[-1] if state.get("rewritten_queries") else state.get("query", "")
    trail = state.get("thought_trail", [])
    
    results = await retriever.retrieve(query=query, top_k=5)
    
    docs = []
    total_saved_chars = 0
    for r in results:
        orig_len = r.metadata.get("original_length", len(r.text))
        comp_len = r.metadata.get("compressed_length", len(r.text))
        total_saved_chars += max(0, orig_len - comp_len)

        docs.append({
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "text": r.text,
            "filename": r.metadata.get("filename", "unknown"),
            "section": r.metadata.get("section", "General"),
            "score": r.rerank_score or r.rrf_score
        })

    compression_note = f" (Context compression saved ~{total_saved_chars} characters)" if total_saved_chars > 0 else ""
    thought = f"RAGSpecialist: Retrieved {len(docs)} high-fidelity chunks using Hybrid RRF search & HyDE expansion for '{query}'{compression_note}."
    return {
        "retrieved_docs": docs,
        "thought_trail": trail + [thought]
    }
