import pytest
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace
from decimal import Decimal
import httpx

from src.evals.benchmark import AMBIGUOUS_TEST_ID, DATA, calibration_agreement, call_diagnostics, completion_estimate, question_tokens, relevant_chunks, reviewed_test_cases, summarize, write_report, run
from src.evals.judging import Answer, Judgement
from src.rag.document_store import DocumentStore
from src.rag.models import DocumentChunk


def test_review_gate_and_span_aware_gold_chunks():
    reviews = [json.loads(line) for line in (DATA / "test_review.jsonl").read_text(encoding="utf-8").splitlines()]
    reviews[0]["reviewed"] = None
    with TemporaryDirectory() as folder:
        path = Path(folder) / "review.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in reviews), encoding="utf-8")
        with pytest.raises(ValueError, match="pending human reviews"):
            reviewed_test_cases(path)
    chunks = [DocumentChunk(id=name, document_id="source", chunk_index=index, text="text",
                            metadata={"start_offset": start, "end_offset": end})
              for index, (name, start, end) in enumerate((("cut", 0, 5), ("support", 2, 10)))]
    assert relevant_chunks({"source_chunk_id": "source", "answers": [{"answer_start": 4, "text": "answer"}]}, chunks) == {"support"}


def test_missing_answer_metrics_stay_missing_in_summary():
    rows = [{"configuration": "test", "split": "dev", "metrics": {"recall@5": 1},
             "answer": None, "judgement": None, "error": None, "cost_usd": "0",
             "retrieval_latency_ms": 12}]
    result = summarize(rows)[0]
    assert result["faithfulness"] is None
    assert result["valid_judgements"] == 0
    assert result["recall@5"] == 1
    assert calibration_agreement([])["agreement"] is None


def test_summary_separates_provider_queue_wall_and_counts_truncations():
    rows = [{"configuration": "test", "split": "dev", "metrics": {}, "cost_usd": "0",
             "answer_provider_latency_ms": 250, "answer_queue_latency_ms": 11000,
             "answer_wall_latency_ms": 11250}]
    summary = summarize(rows)[0]
    assert summary["answer_provider_p50_ms"] == 250
    assert summary["answer_queue_p50_ms"] == 11000
    assert summary["answer_wall_p50_ms"] == 11250
    diagnostic = call_diagnostics([{"finish_reason": "length", "truncated": True, "http_status": 200},
                                   {"truncated": True, "http_status": 400}, {"http_status": 429}])
    assert diagnostic == {"calls": 3, "finish_reason_recorded": 1, "finish_reason_length": 1,
                          "truncations": 2, "http_400s": 1, "http_429s": 1}


def test_generated_table_does_not_round_small_paid_costs_to_zero():
    rows = [{"configuration": "test", "split": "dev", "metrics": {"recall@5": 1},
             "answer": None, "judgement": None, "error": None, "cost_usd": "0.000003",
             "retrieval_latency_ms": 12}]
    report = {"status": "measured", "calibration": {"status": "pending_human_labels"}, "summaries": summarize(rows)}
    with TemporaryDirectory() as folder, patch("src.evals.benchmark.RESULTS", Path(folder)):
        write_report(report, "test")
        assert "0.00000300" in (Path(folder) / "test.md").read_text(encoding="utf-8")
        saved = json.loads((Path(folder) / "test.json").read_text(encoding="utf-8"))
        assert saved["summaries"][0]["provider_cost_usd_per_query"] == 0.000003


def test_ambiguous_test_sensitivity_uses_same_predictions_without_affecting_dev():
    rows = [{"configuration": "test", "split": split, "question_id": question_id,
             "metrics": {"recall@5": score}, "cost_usd": "0"}
            for split, question_id, score in (("dev", "dev-id", 0.5),
                                              ("test", AMBIGUOUS_TEST_ID, 0), ("test", "other", 1))]
    summaries = {row["split"]: row for row in summarize(rows)}
    assert summaries["dev"]["recall@5"] == 0.5
    assert summaries["test"]["questions"] == 2 and summaries["test"]["recall@5"] == 0.5
    assert summaries["test_without_ambiguous"]["questions"] == 1
    assert summaries["test_without_ambiguous"]["recall@5"] == 1
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_resume_keeps_answer_after_judge_interruption_and_retains_costs():
    clients = []

    def factory(*args, **kwargs):
        client = SimpleNamespace(model="test", input_price=Decimal('0.1'), output_price=Decimal('0.2'),
                                 request_interval_seconds=0, max_rate_limit_retries=0, wait_until=None,
                                 spent=Decimal(0), calls=[], close=AsyncMock(), _lock=asyncio.Lock(), role=kwargs.get('role', 'answer'))
        clients.append(client)
        return client

    def record_call(client):
        client.spent += Decimal('0.001')
        client.calls.append({'cost_usd': '0.001', 'client_role': client.role, 'provider_latency_ms': 100, 'queue_wait_ms': 10})
        client.checkpoint()

    async def answer(client, *args):
        record_call(client)
        return Answer(claims=[], abstained=True)

    interrupted = True

    async def judge(client, *args):
        nonlocal interrupted
        record_call(client)
        if interrupted:
            interrupted = False
            raise httpx.ReadTimeout('interrupted')
        return Judgement(answer_relevant=False, claim_support=[], citation_support=[], reason='abstained')

    with TemporaryDirectory() as folder, patch('src.evals.benchmark.RESULTS', Path(folder)), \
         patch('src.evals.benchmark.LocalEmbeddings', return_value=SimpleNamespace(model_name='test', revision='test')), \
         patch('src.evals.benchmark.build_indexes', new=AsyncMock(return_value={mode: DocumentStore() for mode in ('fixed', 'recursive', 'semantic')})), \
         patch('src.evals.benchmark.HybridRetriever'), patch('src.evals.benchmark.LLMClient', side_effect=factory), \
         patch('src.evals.benchmark.retrieve_case', new=AsyncMock(return_value=([], None))), \
         patch('src.evals.benchmark.generate_answer', new=AsyncMock(side_effect=answer)) as generation, \
         patch('src.evals.benchmark.judge_answer', new=AsyncMock(side_effect=judge)), \
         patch('src.evals.benchmark.importlib.metadata.version', return_value='test'), \
         patch('src.evals.benchmark.settings.EVAL_JUDGE_BASE_URL', 'https://judge.example/v1'), \
         patch('src.evals.benchmark.settings.EVAL_JUDGE_MODEL', 'test-judge'):
        args = SimpleNamespace(split='dev', smoke=True, resume=False, prepare_labels=False, retrieval_only=False)
        with pytest.raises(httpx.ReadTimeout):
            await run(args)
        assert generation.await_count == 1
        args.resume = True
        await run(args)
        assert generation.await_count == 5
        assert sum(client.spent for client in clients[-2:]) == Decimal('0.011')
        assert clients[-2].spent == Decimal('0.005')
        assert clients[-1].spent == Decimal('0.006')
        saved = json.loads(next(Path(folder).glob('smoke_*.json')).read_text(encoding='utf-8'))
        assert saved['records'][0]['judge_provider_latency_ms'] == 200
        assert saved['records'][0]['judge_queue_latency_ms'] == 20
        await run(args)
        assert generation.await_count == 5
        args.resume = False
        with pytest.raises(ValueError, match='already exist'):
            await run(args)


def test_per_question_tokens_and_quota_cost_projection_keep_unknown_usage_visible():
    calls = [{'client_role': 'answer', 'prompt_tokens': 600, 'completion_tokens': 200, 'cost_usd': '0.001'},
             {'client_role': 'judge', 'prompt_tokens': 1000, 'completion_tokens': 100, 'cost_usd': '0.003'},
             {'client_role': 'judge', 'prompt_tokens': None, 'completion_tokens': None, 'cost_usd': '0.02'}]
    usage = question_tokens(calls)
    assert usage['answer']['total_tokens'] == 800
    assert usage['judge']['total_tokens'] == 1100 and usage['judge']['unknown_usage_attempts'] == 1
    estimate = completion_estimate([{'completed': True, 'answer': {}, 'judgement': {}, 'token_usage': usage,
                                    'cost_usd': '0.024'}], calls, 501, 200000, Decimal('2.00'))
    assert estimate['remaining_configuration_questions'] == 500
    assert estimate['groq_quota_days'] == 2
    assert Decimal(estimate['projected_remaining_cost_usd']) == 12
    assert estimate['budget_sufficient'] is False
