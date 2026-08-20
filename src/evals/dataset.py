from typing import List, Dict, Any

EVALUATION_DATASET: List[Dict[str, Any]] = [
    {
        "query": "How does LangGraph multi-agent architecture handle supervisor routing and reflection?",
        "ground_truth": "LangGraph uses stateful cyclic graphs where a supervisor classifies intent and routes queries to specialist agents, while reflection graders verify context faithfulness.",
        "expected_domain": "architecture"
    },
    {
        "query": "What are the advantages of pgvector HNSW indexing combined with BM25 via RRF?",
        "ground_truth": "pgvector HNSW provides fast sub-10ms dense cosine similarity search while BM25 handles exact keyword matches. Reciprocal Rank Fusion with k=60 fuses both candidate lists to maximize domain recall.",
        "expected_domain": "storage"
    },
    {
        "query": "How does Model Context Protocol execute tools securely over JSON-RPC 2.0?",
        "ground_truth": "MCP defines standardized JSON-RPC 2.0 tools/list and tools/call methods with parameter schemas, and servers enforce read-only safety guards on SQL queries.",
        "expected_domain": "mcp"
    },
    {
        "query": "What does the query rewriter do when low retrieval relevance is detected?",
        "ground_truth": "When the reflection grader detects low context relevance or potential hallucination, the query rewriter reformulates search terms and loops back to retrieval up to max iterations.",
        "expected_domain": "agents"
    }
]
