import pytest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.evals.benchmark import DATA, calibration_agreement, relevant_chunks, reviewed_test_cases, summarize
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
