import json
from unittest.mock import AsyncMock, patch
from decimal import Decimal

import httpx
import pytest

from src.llm import LLMClient, quota_delay
from pathlib import Path
from tempfile import TemporaryDirectory


def test_token_quota_windows_and_single_request_ceiling():
    events = [{"time": 100, "tokens": 6000}, {"time": 130, "tokens": 1000}]
    assert quota_delay(events, 2000, 140, 8000, 200000) == 20
    assert quota_delay(events, 2000, 160, 8000, 200000) == 0
    assert quota_delay(events, 2000, 140, 8000, 8000) == 86360
    with pytest.raises(ValueError, match="upper bound"):
        quota_delay([], 8001, 140, 8000, 200000)


@pytest.mark.asyncio
async def test_quota_is_saved_before_dispatch_and_reconciled_across_restart():
    with TemporaryDirectory() as folder:
        path = Path(folder) / "quota.json"

        def respond(request):
            pending = json.loads(path.read_text())
            assert pending[-1]["usage"] == "reserved"
            return httpx.Response(200, headers={"x-ratelimit-remaining-tokens": "7985"},
                                  json={"usage": {"prompt_tokens": 10, "completion_tokens": 5},
                                        "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]})

        client = LLMClient("http://localhost/v1", "test", tokens_per_minute=8000, tokens_per_day=200000,
                           quota_path=path, transport=httpx.MockTransport(respond))
        await client.complete([{"role": "user", "content": "hello"}])
        await client.close()
        restarted = LLMClient("http://localhost/v1", "test", quota_path=path)
        try:
            assert restarted.quota_events[-1]["tokens"] == 15
            assert restarted.quota_events[-1]["usage"] == "measured"
            assert client.calls[0]["rate_limit_headers"]["x-ratelimit-remaining-tokens"] == "7985"
        finally:
            await restarted.close()


@pytest.mark.asyncio
async def test_rate_limit_backoff_keeps_every_attempt_in_shared_budget():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(429, headers={"retry-after": "7"},
                                  json=[{"error": {"message": "Quota refused for test-key", "status": "RESOURCE_EXHAUSTED"}}])
        if len(requests) == 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"usage": {"prompt_tokens": 10, "completion_tokens": 5},
                                        "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]})

    client = LLMClient("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash",
                       "test-key", local=False, input_usd_per_million="0.30", output_usd_per_million="2.50",
                       transport=httpx.MockTransport(respond))
    try:
        with patch("src.llm.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await client.complete([{"role": "user", "content": "hello"}])
        assert [call.args[0] for call in sleep.await_args_list] == [7, 4]
        assert requests[0]["reasoning_effort"] == "none"
        assert len(client.calls) == 3
        assert client.calls[0]["http_status"] == 429 and client.calls[0]["backoff_seconds"] == 7
        assert client.calls[0]["cost_kind"] == "reserved_upper_bound"
        assert "test-key" not in json.dumps(client.calls)
        assert client.calls[0]["provider_error_status"] == "RESOURCE_EXHAUSTED"
        assert client.spent == sum(Decimal(row["cost_usd"]) for row in client.calls)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_rate_limit_retries_stop_at_configured_ceiling():
    client = LLMClient("http://localhost/v1", "test", max_rate_limit_retries=2,
                       transport=httpx.MockTransport(lambda request: httpx.Response(429)))
    try:
        with patch("src.llm.asyncio.sleep", new_callable=AsyncMock), pytest.raises(httpx.HTTPStatusError):
            await client.complete([{"role": "user", "content": "hello"}])
        assert len(client.calls) == 3
        assert [row.get("backoff_seconds") for row in client.calls] == [2, 4, None]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_daily_quota_stops_without_repeating_requests():
    response = {"error": {"message": "Daily quota exceeded", "status": "RESOURCE_EXHAUSTED", "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "59707s"}]}}
    client = LLMClient("http://localhost/v1", "test",
                       transport=httpx.MockTransport(lambda request: httpx.Response(429, json=response)))
    try:
        with patch("src.llm.asyncio.sleep", new_callable=AsyncMock) as sleep, pytest.raises(RuntimeError, match="Daily Gemini quota"):
            await client.complete([{"role": "user", "content": "hello"}])
        sleep.assert_not_awaited()
        assert len(client.calls) == 1
        assert client.calls[0]["provider_retry_delay"] == "59707s"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_local_tokens_are_logged_and_paid_budget_blocks_before_dispatch():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"usage": {"prompt_tokens": 10, "completion_tokens": 5},
                                        "choices": [{"finish_reason": "stop", "message": {"content": '{"ok": true}'}}]})

    transport = httpx.MockTransport(respond)
    local = LLMClient("http://127.0.0.1:8080/v1", "local-test", transport=transport)
    try:
        assert json.loads(await local.complete([{"role": "user", "content": "hello"}])) == {"ok": True}
        assert local.calls[0]["prompt_tokens"] == 10
        assert local.calls[0]["completion_tokens"] == 5
        assert local.calls[0]["cost_usd"] == "0"
    finally:
        await local.close()
    remote = LLMClient("https://example.invalid/v1", "priced-test", "test-key", local=False,
                       input_usd_per_million="1000", output_usd_per_million="1000", transport=transport)
    try:
        with pytest.raises(RuntimeError, match="budget exhausted"):
            await remote.complete([{"role": "user", "content": "hello"}])
        assert len(requests) == 1
    finally:
        await remote.close()


@pytest.mark.asyncio
async def test_missing_usage_does_not_become_a_measured_zero():
    client = LLMClient("http://localhost:8080/v1", "test", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": []})))
    try:
        with pytest.raises(ValueError, match="token usage"):
            await client.complete([{"role": "user", "content": "hello"}])
        assert client.calls[0]["status"] == "failed"
        assert client.calls[0]["cost_kind"] == "reserved_upper_bound"
        assert client.calls[0]["prompt_tokens"] is None
    finally:
        await client.close()


def test_zero_cost_cannot_be_claimed_for_remote_endpoint():
    with pytest.raises(ValueError, match="loopback"):
        LLMClient("https://example.invalid/v1", "test")
    with pytest.raises(ValueError, match="below"):
        LLMClient("http://localhost:8080/v1", "test", budget_usd="3")


@pytest.mark.asyncio
async def test_failed_remote_calls_keep_reservations_and_cannot_retry_past_cap():
    requests = []

    def fail(request):
        requests.append(request)
        return httpx.Response(500, json={"error": "unknown provider usage"})

    client = LLMClient("https://example.invalid/v1", "priced-test", "test-key", local=False,
                       budget_usd="0.002", input_usd_per_million="0.1", output_usd_per_million="0.1",
                       transport=httpx.MockTransport(fail), max_rate_limit_retries=0)
    try:
        for _ in range(2):
            with pytest.raises(httpx.HTTPStatusError):
                await client.complete([{"role": "user", "content": "hello"}])
        with pytest.raises(RuntimeError, match="before dispatch"):
            await client.complete([{"role": "user", "content": "hello"}])
        assert len(requests) == 2
        assert client.spent > 0
        assert client.spent < client.budget
        assert all(row["cost_kind"] == "reserved_upper_bound" for row in client.calls)
    finally:
        await client.close()
