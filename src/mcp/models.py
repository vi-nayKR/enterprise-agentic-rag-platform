from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid

class MCPRequest(BaseModel):
    """JSON-RPC 2.0 Request schema."""
    jsonrpc: str = "2.0"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)

class MCPResponse(BaseModel):
    """JSON-RPC 2.0 Response schema."""
    jsonrpc: str = "2.0"
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class ToolDefinition(BaseModel):
    """Schema representing an MCP Tool Definition."""
    name: str
    description: str
    inputSchema: Dict[str, Any]

class ToolCallResult(BaseModel):
    """Standardized tool execution output."""
    content: List[Dict[str, Any]] = Field(default_factory=list)
    is_error: bool = False
