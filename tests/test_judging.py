import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError

from src.evals.judging import Answer, Judgement, answer_metrics, human_label_sheet, judge_answer, prompt_evidence


def test_abstention_is_not_a_perfect_faithfulness_score():
    answer = Answer(claims=[], abstained=True)
    judgement = Judgement(answer_relevant=False, claim_support=[], citation_support=[], reason="No answer")
    assert answer_metrics(answer, judgement)["faithfulness"] is None
    assert answer_metrics(answer, judgement)["answer_relevance"] == 0
    with pytest.raises(ValidationError):
        Answer(claims=[], abstained=False)


def test_compact_prompt_preserves_citation_evidence_title_and_original_metadata():
    chunks = [{'chunk_id': 'source:1', 'text': 'Exact evidence.',
               'metadata': {'title': 'Article', 'start_offset': 50, 'end_offset': 65}}]
    original = json.dumps(chunks)
    assert prompt_evidence(chunks) == [{'chunk_id': 'source:1', 'text': 'Exact evidence.', 'title': 'Article'}]
    assert json.dumps(chunks) == original


@pytest.mark.asyncio
async def test_judge_missing_ids_and_incomplete_labels_are_not_accepted():
    class Client:
        async def complete(self, messages, **kwargs):
            return json.dumps({"answer_relevant": True, "claim_support": [{"claim_index": 0, "supported": True}],
                               "citation_support": [{"claim_index": 0, "citation_id": "invented", "supported": True}],
                               "reason": "stub"})

    answer = Answer(claims=[{"text": "The sky is blue", "citation_ids": ["invented"]}], abstained=False)
    judgement = await judge_answer(Client(), "Sky colour?", ["blue"], [], answer)
    assert answer_metrics(answer, judgement)["citation_accuracy"] == 0
    answer.claims[0].citation_ids.append("missing")
    with pytest.raises(ValueError, match="citation labels"):
        await judge_answer(Client(), "Sky colour?", ["blue"], [], answer)


def test_label_sheet_never_prefills_labels_or_overwrites_review():
    records = [{"id": str(index), "question": "Sky colour?", "reference_answers": ["blue"],
                "contexts": [{"chunk_id": "a", "text": "The sky is blue"}],
                "answer": {"claims": [{"text": "blue", "citation_ids": ["a"]}], "abstained": False}}
               for index in range(20)]
    with TemporaryDirectory() as folder:
        path = Path(folder) / "labels_todo.jsonl"
        human_label_sheet(records, path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert all(row["labels"]["answer_relevant"] is None for row in rows)
        assert all(label["supported"] is None for row in rows for label in row["labels"]["claim_support"])
        rows[0]["reviewer"] = "human"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        original = path.read_bytes()
        human_label_sheet(records, path)
        assert path.read_bytes() == original
        records[0]["answer"]["claims"][0]["text"] = "Different answer"
        with pytest.raises(ValueError, match="different evidence"):
            human_label_sheet(records, path)
