import json
from pathlib import Path

from scripts.prepare_squad import build_subset

DATA = Path(__file__).resolve().parents[1] / "data" / "squad_v1"


def test_committed_squad_split_and_spans():
    corpus = [json.loads(line) for line in (DATA / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {row["id"]: row for row in corpus}
    assert len(by_id) == len(corpus)
    examples = {}
    for split, count in (("dev", 60), ("test", 40)):
        rows = [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == count
        assert len({row["id"] for row in rows}) == count
        for row in rows:
            assert row["split"] == split and row["synthetic"] is False
            passage = by_id[row["source_chunk_id"]]
            assert passage["split"] == split
            for answer in row["answers"]:
                start = answer["answer_start"]
                assert passage["text"][start:start + len(answer["text"])] == answer["text"]
        examples[split] = rows
    assert {row["article"] for row in examples["dev"]}.isdisjoint(row["article"] for row in examples["test"])
    assert {row["text"] for row in corpus if row["split"] == "dev"}.isdisjoint(
        row["text"] for row in corpus if row["split"] == "test")


def test_subset_rejects_missing_source_articles():
    try:
        build_subset({"data": []})
    except ValueError as error:
        assert "Missing approved articles" in str(error)
    else:
        raise AssertionError("Missing approved articles must fail rather than change the subset")
