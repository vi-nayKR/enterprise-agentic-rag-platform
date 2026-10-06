"""One OpenAI-compatible interface for local or explicitly priced remote models."""

import asyncio
import json
import time
import math
from pathlib import Path
from decimal import Decimal
from urllib.parse import urlparse

import httpx


def quota_delay(events: list[dict], tokens: int, now: float, minute_limit: int, day_limit: int) -> float:
    """Conservative rolling windows; pending/unknown usage keeps its reservation."""
    delay = 0.0
    for window, limit in ((60, minute_limit), (86400, day_limit)):
        if not limit:
            continue
        if tokens > limit:
            raise ValueError("One request's token upper bound exceeds the configured quota")
        active = sorted((row for row in events if row["time"] + window > now), key=lambda row: row["time"])
        total = sum(row["tokens"] for row in active)
        for row in active:
            if total + tokens <= limit:
                break
            total -= row["tokens"]
            delay = max(delay, row["time"] + window - now)
    return delay


class LLMClient:
    def __init__(
        self, base_url: str, model: str, api_key: str = "", *, local: bool = True,
        budget_usd: str = "2.99", input_usd_per_million: str = "0",
        output_usd_per_million: str = "0", input_token_limit: int = 8192,
        transport: httpx.AsyncBaseTransport | None = None,
        max_rate_limit_retries: int = 6, request_interval_seconds: float = 0,
        tokens_per_minute: int = 0, tokens_per_day: int = 0, quota_path: Path | None = None,
    ):
        self.model = model
        self._api_key = api_key
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
        if max_rate_limit_retries < 0 or request_interval_seconds < 0:
            raise ValueError("Retry count and request interval must be nonnegative")
        self.max_rate_limit_retries = max_rate_limit_retries
        self.request_interval_seconds = request_interval_seconds
        self._last_dispatch = 0.0
        self._gemini = urlparse(base_url).hostname == "generativelanguage.googleapis.com"
        self._groq = urlparse(base_url).hostname == "api.groq.com"
        if min(tokens_per_minute, tokens_per_day) < 0:
            raise ValueError("Token quotas must be nonnegative")
        self.tokens_per_minute = tokens_per_minute
        self.tokens_per_day = tokens_per_day
        self.quota_path = quota_path
        # ponytail: one evaluator owns this ledger; add a file lock for concurrent runs.
        self.quota_events = json.loads(quota_path.read_text(encoding="utf-8")) if quota_path and quota_path.exists() else []
        self.checkpoint = None
        self.wait_until = None
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

    def _save_quota(self) -> None:
        if self.quota_path:
            self.quota_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.quota_path.with_suffix(".part")
            temporary.write_text(json.dumps(self.quota_events), encoding="utf-8")
            temporary.replace(self.quota_path)
        if self.checkpoint:
            self.checkpoint()

    async def _wait(self, seconds: float, reason: str) -> None:
        self.wait_until = time.time() + seconds
        print(f"{self.model}: {reason}; waiting {seconds:.1f}s", flush=True)
        if self.checkpoint:
            self.checkpoint()
        while seconds > 0:
            part = min(seconds, 60)
            await asyncio.sleep(part)
            seconds -= part
        self.wait_until = None

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int = 256,
                       schema: dict | None = None) -> str:
        for attempt in range(self.max_rate_limit_retries + 1):
            try:
                return await self._complete_once(messages, max_tokens=max_tokens, schema=schema)
            except httpx.HTTPStatusError as error:
                if error.response.status_code == 429 and self.calls[-1].get("quota_exhausted") == "daily":
                    raise RuntimeError("Daily Gemini quota exhausted; wait for reset or use a key with paid access") from error
                if error.response.status_code not in (429, 500, 502, 503, 504) or attempt == self.max_rate_limit_retries:
                    raise
                delay = min(60, 2 ** (attempt + 1))
                retry_after = error.response.headers.get("retry-after", "")
                try:
                    value = float(retry_after)
                    if math.isfinite(value):
                        delay = max(delay, value)
                except ValueError:
                    pass
                self.calls[-1].update(retry_attempt=attempt + 1, backoff_seconds=delay)
                print(f"{self.model}: HTTP {error.response.status_code}; retry {attempt + 1} after {delay:g}s", flush=True)
                await self._wait(delay, "provider backoff")

    async def _complete_once(self, messages: list[dict[str, str]], *, max_tokens: int,
                             schema: dict | None) -> str:
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
            remaining = self.request_interval_seconds - (time.monotonic() - self._last_dispatch)
            while remaining > 0:
                await asyncio.sleep(min(remaining, 60))
                remaining = self.request_interval_seconds - (time.monotonic() - self._last_dispatch)
            if self.tokens_per_minute or self.tokens_per_day:
                token_bound = len(json.dumps(messages, ensure_ascii=False).encode()) + 256 + max_tokens
                if schema:
                    token_bound += len(json.dumps(schema, ensure_ascii=False).encode())
                while delay := quota_delay(self.quota_events, token_bound, time.time(), self.tokens_per_minute, self.tokens_per_day):
                    await self._wait(delay + 0.01, "token quota")
                self.quota_events = [row for row in self.quota_events if row["time"] + 86400 > time.time()]
                event = {"time": time.time(), "tokens": token_bound, "usage": "reserved"}
                self.quota_events.append(event)
            else:
                event = None
            self.spent += reserve
            record = {"model": self.model, "status": "dispatched", "prompt_tokens": None,
                      "completion_tokens": None, "cost_usd": str(reserve), "cost_kind": "reserved_upper_bound"}
            self.calls.append(record)
            self._save_quota()
            try:
                self._last_dispatch = time.monotonic()
                body = {
                    "model": self.model, "messages": messages, "temperature": 0,
                    "max_tokens": max_tokens, "response_format": (
                        {"type": "json_schema", "json_schema": {"name": "evidencerag", "strict": True, "schema": schema}}
                        if schema is not None else {"type": "json_object"}),
                }
                # Flash's default thinking can consume the bounded output allowance.
                if self._gemini and "flash" in self.model and self.model.startswith("gemini-2.5"):
                    body["reasoning_effort"] = "none"
                elif self._gemini and self.model.startswith("gemini-3"):
                    body["reasoning_effort"] = "low"
                elif self._groq and self.model.startswith("openai/gpt-oss"):
                    body["reasoning_effort"] = "low"
                response = await self._http.post("chat/completions", json=body)
                record["http_status"] = response.status_code
                record["rate_limit_headers"] = {name: value for name, value in response.headers.items()
                                                if name.startswith("x-ratelimit-") or name == "retry-after"}
                if not response.is_success:
                    try:
                        detail = response.json()
                        if isinstance(detail, list):
                            detail = detail[0]
                        detail = detail.get("error", {})
                        message = str(detail.get("message", ""))
                        record["provider_error_message"] = message.replace(self._api_key, "<redacted>") if self._api_key else message
                        record["provider_error_status"] = detail.get("status")
                        for item in detail.get("details", []):
                            if any("PerDay" in violation.get("quotaId", "") for violation in item.get("violations", [])):
                                record["quota_exhausted"] = "daily"
                            if item.get("@type", "").endswith("RetryInfo"):
                                record["provider_retry_delay"] = item.get("retryDelay")
                    except (ValueError, IndexError, AttributeError):
                        pass
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
                if event is not None:
                    event.update(tokens=prompt + completion, usage="measured")
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
            finally:
                self._save_quota()
