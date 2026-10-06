"""Build the approved human-authored SQuAD subset without generating questions."""

import hashlib
import json
import random
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "squad_v1"
URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
SOURCE_SHA256 = "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"
ARTICLES = {
    "Computational_complexity_theory": "dev",
    "Packet_switching": "dev",
    "Steam_engine": "dev",
    "Nikola_Tesla": "dev",
    "Oxygen": "dev",
    "Apollo_program": "test",
    "Prime_number": "test",
    "Force": "test",
    "Chloroplast": "test",
    "Geology": "test",
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build_subset(source: dict) -> tuple[list[dict], list[dict]]:
    articles = {article["title"]: article for article in source["data"]}
    missing = ARTICLES.keys() - articles.keys()
    if missing:
        raise ValueError(f"Missing approved articles: {sorted(missing)}")
    passages, examples = [], []
    rng = random.Random(2026)
    for title, split in ARTICLES.items():
        candidates = []
        for paragraph in articles[title]["paragraphs"]:
            text = paragraph["context"]
            passage_id = hashlib.sha256((title + "\n" + text).encode()).hexdigest()[:16]
            passages.append({"id": passage_id, "title": title, "text": text, "split": split})
            if paragraph["qas"]:
                question = rng.choice(sorted(paragraph["qas"], key=lambda qa: qa["id"]))
                for answer in question["answers"]:
                    start = answer["answer_start"]
                    if start < 0 or text[start:start + len(answer["text"])] != answer["text"]:
                        raise ValueError(f"Invalid answer span: {question['id']}")
                candidates.append({
                    "id": question["id"], "question": question["question"],
                    "answers": question["answers"], "source_chunk_id": passage_id,
                    "article": title, "split": split, "origin": "human_authored_squad",
                    "synthetic": False,
                })
        examples.extend(rng.sample(candidates, 12 if split == "dev" else 8))
    return passages, examples


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    upstream = DEST / "upstream-dev-v1.1.json"
    if not upstream.exists():
        with urlopen(URL, timeout=60) as response:
            upstream.write_bytes(response.read())
    raw = upstream.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Upstream content changed; create a new dataset version rather than overwrite squad_v1")
    passages, examples = build_subset(json.loads(raw))
    write_jsonl(DEST / "corpus.jsonl", passages)
    for split in ("dev", "test"):
        write_jsonl(DEST / f"{split}.jsonl", [row for row in examples if row["split"] == split])
    # Never overwrite the user's corrections or review status on regeneration.
    review = DEST / "test_review.jsonl"
    if not review.exists():
        texts = {row["id"]: row["text"] for row in passages}
        write_jsonl(review, [dict(row, source_text=texts[row["source_chunk_id"]],
                                 reviewed=None, reviewer=None, corrected_answers=None, notes=None)
                            for row in examples if row["split"] == "test"])
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    report = {
        "dataset": "SQuAD 1.1", "version": "squad_v1", "source_url": URL,
        "source_sha256": hashlib.sha256(raw).hexdigest(), "license": "CC-BY-SA-4.0",
        "sampling_seed": 2026, "articles": len(ARTICLES), "passages": len(passages),
        "dev_examples": sum(row["split"] == "dev" for row in examples),
        "test_examples": sum(row["split"] == "test" for row in examples),
        "human_test_review_complete": False,
    }
    (results / "dataset_v1.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (results / "dataset_v1.md").write_text(
        "| Dataset | Articles | Passages | Dev examples | Test examples |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| SQuAD 1.1 subset | {report['articles']} | {report['passages']} | "
        f"{report['dev_examples']} | {report['test_examples']} |\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
