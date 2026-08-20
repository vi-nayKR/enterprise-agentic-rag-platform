from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid


class Document(BaseModel):
    """Represents a raw ingested document."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    content_type: str = "text/plain"
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentChunk(BaseModel):
    """Represents a chunk extracted from a document with vector & text representations."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    chunk_index: int
    text: str
    token_count: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchQuery(BaseModel):
    """Encapsulates a user search query and retrieval parameters."""

    query: str
    top_k: int = 5
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Unified search result combining dense, sparse, RRF, and reranker scores."""

    chunk_id: str
    document_id: str
    text: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)