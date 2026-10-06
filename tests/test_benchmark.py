import pytest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.evals.benchmark import AMBIGUOUS_TEST_ID, DATA, calibration_agreement, relevant_chunks, reviewed_test_cases, summarize, write_report
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
