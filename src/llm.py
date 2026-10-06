"""One OpenAI-compatible interface for local or explicitly priced remote models."""

import asyncio
import json
from decimal import Decimal
from urllib.parse import urlparse

import httpx


class LLMClient:
    def __init__(
        self, base_url: str, model: str, api_key: str = "", *, local: bool = True,
        budget_usd: str = "2.99", input_usd_per_million: str = "0",
        output_usd_per_million: str = "0", input_token_limit: int = 8192,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model
        self.budget = Decimal(budget_usd)
        self.input_price = Decimal(input_usd_per_million)
        self.output_price = Decimal(output_usd_per_million)
        prices = (self.budget, self.input_price, self.output_price)
        if any(not value.is_finite() for value in prices) or not 0 < self.budget < 3:
            raise ValueError("Full-run budget must be finite, positive, and strictly below $3")
        if input_token_limit < 1 or min(self.input_price, self.output_price) < 0:
            raise ValueError("Require a positive token limit and nonnegative prices")
        if local:
            if urlparse(base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("Zero-cost local mode requires a loopback endpoint")
            if self.input_price or self.output_price:
                raise ValueError("Local requests must not have provider token charges")
        elif not api_key or not self.input_price or not self.output_price:
            raise ValueError("Remote calls require credentials and explicit positive token prices")
        self.input_token_limit = input_token_limit
        self.spent = Decimal(0)
        self.calls: list[dict] = []
        self._lock = asyncio.Lock()
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/", timeout=180,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            transport=transport,
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int = 256,
                       schema: dict | None = None) -> str:
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Output token limit must be positive")
        # UTF-8 bytes plus framing allowance bound input length conservatively;
        # reserve the entire configured input-token ceiling, not an optimistic estimate.
        if len(json.dumps(messages, ensure_ascii=False).encode()) + 256 > self.input_token_limit:
            raise ValueError("Prompt exceeds the conservative input-token ceiling")
        reserve = (self.input_token_limit * self.input_price + max_tokens * self.output_price) / 1_000_000
        async with self._lock:
            if self.spent + reserve > self.budget:
                raise RuntimeError("Evaluation budget exhausted before dispatch")
            self.spent += reserve
            record = {"model": self.model, "status": "dispatched", "prompt_tokens": None,
                      "completion_tokens": None, "cost_usd": str(reserve), "cost_kind": "reserved_upper_bound"}
            self.calls.append(record)
            try:
                response = await self._http.post("chat/completions", json={
                    "model": self.model, "messages": messages, "temperature": 0,
                    "max_tokens": max_tokens, "response_format": (
                        {"type": "json_schema", "json_schema": {"name": "evidencerag", "strict": True, "schema": schema}}
                        if schema is not None else {"type": "json_object"}),
                })
                response.raise_for_status()
                payload = response.json()
                usage = payload.get("usage", {})
                prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
                if type(prompt) is not int or type(completion) is not int or min(prompt, completion) < 0:
                    raise ValueError("Provider omitted valid token usage; no measured-cost claim is possible")
                if prompt > self.input_token_limit or completion > max_tokens:
                    raise ValueError("Provider exceeded the reserved token ceilings")
                actual = (prompt * self.input_price + completion * self.output_price) / 1_000_000
                self.spent += actual - reserve
                record.update(prompt_tokens=prompt, completion_tokens=completion,
                              cost_usd=str(actual), cost_kind="measured", status="completed")
                choice = payload["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("Incomplete model output; do not score a truncated answer")
                text = choice["message"]["content"]
                if not isinstance(text, str):
                    raise ValueError("Missing model text")
                json.loads(text)
                return text
            except Exception:
                record["status"] = "failed"
                raise
