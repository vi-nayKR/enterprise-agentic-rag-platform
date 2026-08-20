import sqlite3
import re
from typing import Dict, Any, List
from src.mcp.models import MCPRequest, MCPResponse

class DatabaseMCPServer:
    """
    Reference MCP Server exposing SQL querying and schema inspection
    over JSON-RPC 2.0 with strict read-only security guards.
    """
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_sample_database()

    def _init_sample_database(self):
        """Populates sample enterprise schema for testing."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_accounts (
                id INTEGER PRIMARY KEY,
                account_name TEXT NOT NULL,
                tier TEXT NOT NULL,
                active_users INTEGER DEFAULT 1,
                monthly_spend REAL DEFAULT 0.0,
                status TEXT DEFAULT 'active'
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY,
                account_id INTEGER,
                event_type TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            )
        """)
        # Sample data
        cursor.execute("DELETE FROM customer_accounts")
        cursor.execute("DELETE FROM audit_logs")
        cursor.executemany("""
            INSERT INTO customer_accounts (id, account_name, tier, active_users, monthly_spend, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            (101, "Acme Global", "Enterprise", 450, 12500.00, "active"),
            (102, "TechCorp LLC", "Pro", 85, 2400.00, "active"),
            (103, "DevStartup Inc", "Starter", 12, 350.00, "suspended"),
            (104, "CloudScale IO", "Enterprise", 1200, 32000.00, "active")
        ])
        cursor.executemany("""
            INSERT INTO audit_logs (id, account_id, event_type, details)
            VALUES (?, ?, ?, ?)
        """, [
            (1, 101, "UPGRADE_TIER", "Upgraded from Pro to Enterprise tier."),
            (2, 103, "PAYMENT_FAILED", "Credit card expired on renewal."),
            (3, 104, "SECURITY_AUDIT", "SOC2 compliance check completed.")
        ])
        self.conn.commit()

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handles incoming JSON-RPC 2.0 requests."""
        if request.method == "tools/list":
            return MCPResponse(
                id=request.id,
                result={
                    "tools": [
                        {
                            "name": "list_tables",
                            "description": "Lists all database tables and column schemas.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {}
                            }
                        },
                        {
                            "name": "query_database",
                            "description": "Executes a read-only SELECT SQL query with security validation.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "sql_query": {
                                        "type": "string",
                                        "description": "The read-only SELECT query to execute"
                                    }
                                },
                                "required": ["sql_query"]
                            }
                        }
                    ]
                }
            )

        elif request.method == "tools/call":
            params = request.params or {}
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if tool_name == "list_tables":
                cursor = self.conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row["name"] for row in cursor.fetchall()]
                
                schema_info = {}
                for t in tables:
                    cursor.execute(f"PRAGMA table_info({t})")
                    schema_info[t] = [{"col": col["name"], "type": col["type"]} for col in cursor.fetchall()]

                return MCPResponse(
                    id=request.id,
                    result={
                        "content": [
                            {"type": "text", "text": f"Tables & Schemas: {schema_info}"}
                        ],
                        "isError": False
                    }
                )

            elif tool_name == "query_database":
                sql = arguments.get("sql_query", "").strip()
                # Security Guard: Block destructive/modifying commands
                forbidden_keywords = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE", "ATTACH", "DETACH"]
                if not sql.upper().startswith("SELECT") or any(re.search(rf"\b{kw}\b", sql, re.IGNORECASE) for kw in forbidden_keywords):
                    return MCPResponse(
                        id=request.id,
                        error={"code": -32602, "message": "Security Error: Only read-only SELECT queries are allowed."}
                    )

                try:
                    cursor = self.conn.cursor()
                    cursor.execute(sql)
                    rows = [dict(row) for row in cursor.fetchall()]
                    return MCPResponse(
                        id=request.id,
                        result={
                            "content": [
                                {"type": "text", "text": f"Query Result ({len(rows)} rows): {rows}"}
                            ],
                            "isError": False
                        }
                    )
                except Exception as e:
                    return MCPResponse(
                        id=request.id,
                        error={"code": -32000, "message": f"Database execution error: {str(e)}"}
                    )

            else:
                return MCPResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Tool '{tool_name}' not found."}
                )

        return MCPResponse(
            id=request.id,
            error={"code": -32600, "message": f"Method '{request.method}' not supported."}
        )
