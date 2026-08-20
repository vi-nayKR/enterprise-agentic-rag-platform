# 📚 Phase 2 Deep-Dive: Model Context Protocol (MCP) Tool Integration

---

## 🗺️ Complete MCP Architecture & Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. AGENT RUNTIME / LANGGRAPH                                │
│    - Needs to execute a database query or fetch API data    │
│    - Invokes `mcp_manager.call_tool(name, arguments)`       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. MCP MANAGER (src/mcp/manager.py)                         │
│    - Maintains registry of connected MCP Clients            │
│    - Maps tool name -> Target MCP Server Client             │
│    - Routes request to Database Client or API Client        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼ (JSON-RPC 2.0: tools/call)
┌─────────────────────────────────────────────────────────────┐
│ 3. ASYNC MCP CLIENT (src/mcp/client.py)                     │
│    - Constructs `MCPRequest(method="tools/call", params)`   │
│    - Transmits request over transport (stdio / SSE / async) │
│    - Validates response schema and captures errors          │
└──────────────────────────────┬──────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│ 4A. DATABASE MCP SERVER     │ │ 4B. ENTERPRISE API SERVER   │
│ (src/mcp/servers/db.py)     │ │ (src/mcp/servers/api.py)    │
│ - list_tables               │ │ - fetch_service_health      │
│ - query_database (read-only)│ │ - search_knowledge_tickets  │
│ - Security Guards (No DROP) │ │ - Returns metrics & tickets │
└─────────────────────────────┘ └─────────────────────────────┘
```

---

## 🛠️ Step 2.1: JSON-RPC 2.0 & Data Schemas (`src/mcp/models.py`)

### 🧠 Why JSON-RPC 2.0?
Anthropic's Model Context Protocol is an open standard designed to eliminate custom, fragile tool connectors. It uses the stateless **JSON-RPC 2.0 specification**:
- Every request has a unique `id`, a standard `jsonrpc: "2.0"` header, a `method`, and `params`.
- Every response returns either a `result` payload or a structured `error` object with error codes (e.g., `-32601` for Tool Not Found, `-32602` for Invalid Params).

---

## 🔌 Step 2.2: Async MCP Client (`src/mcp/client.py`)

### 🧠 Client Responsibilities:
1. **Dynamic Tool Discovery (`tools/list`)**: When initializing, the client queries the server to inspect tool definitions, parameter requirements, and descriptions.
2. **Execution (`tools/call`)**: Packages the agent's proposed arguments into the standard JSON-RPC envelope and returns a `ToolCallResult`.

---

## 🗄️ Step 2.3 & 2.4: Reference MCP Servers

### 🧠 1. Database Server with Read-Only Security Guard (`src/mcp/servers/database_server.py`):
In enterprise RAG, giving an LLM direct database access is dangerous. We implement a strict security validation layer that:
- Ensures queries strictly start with `SELECT`.
- Rejects modifying keywords (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `TRUNCATE`).
- Returns structured rows as JSON strings.

### 🧠 2. Enterprise API Server (`src/mcp/servers/api_server.py`):
Simulates internal microservice health metrics and incident support tickets over MCP.

---

## 🌐 Step 2.5: MCP Manager & Unified Pool (`src/mcp/manager.py`)

### 🧠 The Multi-Server Aggregator Pattern:
Agents should not need to know whether `query_database` belongs to Server A or Server B. `MCPManager`:
1. Connects to all registered MCP servers.
2. Builds an internal routing lookup table (`tool_to_server_map`).
3. Exposes a single `call_tool(name, arguments)` entrypoint.

---

## 🏆 Phase 2 Verification Summary
- **Pytest Suite:** `tests/test_mcp.py`
- **Result:** `5 passed in 0.33s` (100% Pass Rate)
- **Deliverables Created:**
  - `src/mcp/models.py`
  - `src/mcp/client.py`
  - `src/mcp/servers/database_server.py`
  - `src/mcp/servers/api_server.py`
  - `src/mcp/manager.py`
  - `tests/test_mcp.py`
  - `docs/phase2.md`
