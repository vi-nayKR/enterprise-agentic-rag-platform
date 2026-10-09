"""Judge the 20 frozen calibration answers and compute judge-human agreement.

Run after data/squad_v1/labels_todo.jsonl is fully labelled:
    python -m scripts.run_calibration
Uses the configured judge (EVAL_JUDGE_*, falling back to EVAL_LLM_*); about 20 judge calls.
Writes results/judge_calibration.{json,md}. Answers are the frozen originals, never regenerated.
"""

import asyncio
import json
from urllib.parse import urlparse

from src.evals.benchmark import LLMClient, calibrate_judge, settings


def judge_client() -> LLMClient:
    url = settings.EVAL_JUDGE_BASE_URL or settings.EVAL_LLM_BASE_URL
    model = settings.EVAL_JUDGE_MODEL or settings.EVAL_LLM_MODEL
    key = settings.EVAL_JUDGE_API_KEY or settings.EVAL_LLM_API_KEY
    rate_in = (settings.EVAL_JUDGE_INPUT_USD_PER_MILLION
               if settings.EVAL_JUDGE_INPUT_USD_PER_MILLION is not None else settings.EVAL_INPUT_USD_PER_MILLION)
    rate_out = (settings.EVAL_JUDGE_OUTPUT_USD_PER_MILLION
                if settings.EVAL_JUDGE_OUTPUT_USD_PER_MILLION is not None else settings.EVAL_OUTPUT_USD_PER_MILLION)
    return LLMClient(
        url, model, key,
        local=urlparse(url).hostname in {"localhost", "127.0.0.1", "::1"},
        budget_usd=str(settings.EVAL_BUDGET_USD),
        input_usd_per_million=str(rate_in), output_usd_per_million=str(rate_out), role="judge",
        request_interval_seconds=settings.EVAL_JUDGE_REQUEST_INTERVAL_SECONDS,
        max_rate_limit_retries=settings.EVAL_RATE_LIMIT_RETRIES,
        auto_resume_daily_quota=settings.EVAL_AUTO_RESUME_DAILY_QUOTA,
    )


async def main() -> None:
    result = await calibrate_judge(judge_client())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
