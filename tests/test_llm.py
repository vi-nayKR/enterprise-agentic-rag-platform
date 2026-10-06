import json

import httpx
import pytest

from src.llm import LLMClient


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
