"""Reproducible source-aware ablations, with explicit human-review gates."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from config.settings import settings

from src.evals.judging import Answer, answer_metrics, generate_answer, human_label_sheet, judge_answer
from src.evals.ranking_metrics import agreement, percentile, ranking_metrics
from src.llm import LLMClient
from src.rag.chunking import SemanticChunker
from src.rag.document_store import DocumentStore
from src.rag.embeddings import LocalEmbeddings
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.models import Document, DocumentChunk
from src.rag.rrf import reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "squad_v1"
RESULTS = ROOT / "results"
CONFIGS = [
    ("fixed_hybrid", "fixed", "hybrid", False),
    ("recursive_dense", "recursive", "dense", False),
    ("recursive_bm25", "recursive", "bm25", False),
    ("recursive_hybrid", "recursive", "hybrid", False),
    ("recursive_hybrid_rerank", "recursive", "hybrid_rerank", False),
    ("semantic_hybrid", "semantic", "hybrid", False),
    ("recursive_hybrid_rewrite", "recursive", "hybrid", True),
]


class RewrittenQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=500)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reviewed_test_cases(review_path: Path = DATA / "test_review.jsonl") -> list[dict]:
    original = {row["id"]: row for row in read_jsonl(DATA / "test.jsonl")}
    reviews = read_jsonl(review_path)
    if len(reviews) != len(original) or {row["id"] for row in reviews} != original.keys():
        raise ValueError("Human review must cover every original test ID exactly once")
    corpus = {row["id"]: row for row in read_jsonl(DATA / "corpus.jsonl")}
    cases = []
    for review in reviews:
        if review.get("reviewed") is not True or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
            raise ValueError("Test split has pending human reviews; edit test_review.jsonl before held-out evaluation")
        row = dict(original[review["id"]])
        if review["question"] != row["question"] or review["source_chunk_id"] != row["source_chunk_id"]:
            raise ValueError("Question/source changes require a new version, not a silent review-file edit")
        if review.get("corrected_answers") is not None:
            answers = review["corrected_answers"]
            if not isinstance(answers, list) or not answers:
                raise ValueError("Corrected answers must be a nonempty list of text/answer_start records")
            row["answers"] = answers
            row["origin"] = "human_reviewed_squad"
        text = corpus[row["source_chunk_id"]]["text"]
        for answer in row["answers"]:
            start, value = answer["answer_start"], answer["text"]
            if type(start) is not int or start < 0 or not value or text[start:start + len(value)] != value:
                raise ValueError(f"Review correction has an invalid source span: {row['id']}")
        cases.append(row)
    return cases


def relevant_chunks(case: dict, chunks: list[DocumentChunk]) -> set[str]:
    return {chunk.id for chunk in chunks if chunk.document_id == case["source_chunk_id"]
            and any(chunk.metadata["start_offset"] <= answer["answer_start"]
                    and chunk.metadata["end_offset"] >= answer["answer_start"] + len(answer["text"])
                    for answer in case["answers"])}


async def build_indexes(embeddings: LocalEmbeddings) -> dict[str, DocumentStore]:
    corpus = read_jsonl(DATA / "corpus.jsonl")
    fingerprint = {
        "corpus_sha256": hashlib.sha256((DATA / "corpus.jsonl").read_bytes()).hexdigest(),
        "chunker_sha256": hashlib.sha256((ROOT / "src" / "rag" / "chunking.py").read_bytes()).hexdigest(),
        "model": embeddings.model_name, "model_revision": embeddings.revision,
        "sentence_transformers": importlib.metadata.version("sentence-transformers"),
        "chunk_size": 800, "chunk_overlap": 150,
    }
    cache = ROOT / ".cache" / "indexes"
    cache.mkdir(parents=True, exist_ok=True)
    stores = {}
    for mode in ("fixed", "recursive", "semantic"):
        path = cache / f"{mode}.json"
        store = DocumentStore()
        documents = [Document(id=row["id"], filename=row["title"] + ".txt", text=row["text"],
                              metadata={"title": row["title"]}) for row in corpus]
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if saved and saved["fingerprint"] == fingerprint:
            chunks = [DocumentChunk.model_validate(row) for row in saved["chunks"]]
        else:
            chunker = SemanticChunker(mode=mode)
            chunks = []
            for document in documents:
                chunks.extend(await chunker.chunk_document_async(document, embeddings))
            vectors = await embeddings.embed_documents([chunk.text for chunk in chunks])
            if len(vectors) != len(chunks):
                raise ValueError("Embedding count mismatch during index construction")
            for chunk, vector in zip(chunks, vectors):
                chunk.embedding = vector
            temporary = path.with_suffix(".json.part")
            temporary.write_text(json.dumps({"fingerprint": fingerprint,
                                             "chunks": [chunk.model_dump() for chunk in chunks]}), encoding="utf-8")
            temporary.replace(path)
        sources = {document.id: document.text for document in documents}
        for chunk in chunks:
            if chunk.text != sources[chunk.document_id][chunk.metadata["start_offset"]:chunk.metadata["end_offset"]]:
                raise ValueError("Index contains unverifiable source offsets")
            if not chunk.embedding or len(chunk.embedding) != 384:
                raise ValueError("Index contains incompatible embedding vectors")
        for document in documents:
            await store.add_document(document)
        await store.add_chunks(chunks)
        stores[mode] = store
        print(f"Index {mode}: {len(chunks)} source-preserving chunks", flush=True)
    return stores


async def retrieve_case(retriever: HybridRetriever, case: dict, rewrite: bool, client: LLMClient | None):
    results = await retriever.retrieve(case["question"], top_k=10, use_cache=False, use_compression=False)
    rewritten = None
    if rewrite:
        prompt = "Rewrite this question as a concise retrieval query preserving all named entities and intent. "
        text = await client.complete([{"role": "user", "content": prompt + case["question"] + '\nReturn JSON {"query":"..."}.'}],
                                     max_tokens=128, schema=RewrittenQuery.model_json_schema())
        rewritten = RewrittenQuery.model_validate_json(text).query
        additional = await retriever.retrieve(rewritten, top_k=10, use_cache=False, use_compression=False)
        results = reciprocal_rank_fusion(results, additional)[:10]
    return results, rewritten


def calibration_agreement(records: list[dict]) -> dict:
    path = DATA / "labels_todo.jsonl"
    if not path.exists():
        return {"status": "pending_human_labels", "agreement": None}
    labels = read_jsonl(path)
    by_id = {record["id"]: record for record in records}
    if len(labels) != 20 or len({row["id"] for row in labels}) != 20:
        raise ValueError("Calibration requires exactly twenty distinct human-labelled examples")
    pairs = defaultdict(lambda: ([], []))
    for row in labels:
        human = row["labels"]
        values = [human.get("answer_relevant")] + [item.get("supported") for key in ("claim_support", "citation_support")
                                                  for item in human.get(key, [])]
        if not row.get("reviewer") or any(type(value) is not bool for value in values):
            return {"status": "pending_human_labels", "agreement": None}
        record = by_id.get(row["id"])
        if not record or not record.get("judgement"):
            return {"status": "pending_matching_judge_run", "agreement": None}
        evidence = {key: record[key] for key in ("id", "question", "reference_answers", "contexts", "answer")}
        if hashlib.sha256(json.dumps(evidence, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != row["evidence_sha256"]:
            raise ValueError("Human labels do not match the scored evidence")
        judged = record["judgement"]
        pairs["answer_relevance"][0].append(human["answer_relevant"])
        pairs["answer_relevance"][1].append(judged["answer_relevant"])
        for key in ("claim_support", "citation_support"):
            def identity(item):
                return (item["claim_index"], item.get("citation_id"))
            h = {identity(item): item["supported"] for item in human[key]}
            j = {identity(item): item["supported"] for item in judged[key]}
            if h.keys() != j.keys() or len(h) != len(human[key]):
                raise ValueError("Human and judge label identities differ")
            pairs[key][0].extend(h.values())
            pairs[key][1].extend(j[index] for index in h)
    return {"status": "complete", "examples": len(labels),
            "metrics": {key: agreement(*values) if values[0] else None for key, values in pairs.items()}}


async def calibrate_judge(client: LLMClient) -> dict:
    path = DATA / "labels_todo.jsonl"
    if not path.exists():
        return {"status": "pending_human_labels", "agreement": None}
    labels = read_jsonl(path)
    for row in labels:
        values = [row["labels"].get("answer_relevant")] + [item.get("supported") for key in ("claim_support", "citation_support")
                                                         for item in row["labels"].get(key, [])]
        if not row.get("reviewer") or any(type(value) is not bool for value in values):
            return {"status": "pending_human_labels", "agreement": None}
    # Judge the frozen original answers, not regenerated answers that could drift.
    frozen = json.loads((RESULTS / "calibration_inputs.json").read_text(encoding="utf-8"))["records"]
    expected = {row["id"]: row for row in labels}
    if len(expected) != 20 or {row["id"] for row in frozen} != expected.keys():
        raise ValueError("Human labels must match the frozen calibration examples")
    for record in frozen:
        evidence = {key: record[key] for key in ("id", "question", "reference_answers", "contexts", "answer")}
        if hashlib.sha256(json.dumps(evidence, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != expected[record["id"]]["evidence_sha256"]:
            raise ValueError("Human worksheet changed the frozen evidence")
        answer = Answer.model_validate(record["answer"])
        judgement = await judge_answer(client, record["question"], record["reference_answers"], record["contexts"], answer)
        record["judgement"] = judgement.model_dump()
    result = calibration_agreement(frozen)
    result["judge_model"] = client.model
    result["records"] = frozen
    (RESULTS / "judge_calibration.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["| Dimension | Labels | Agreement | Cohen kappa |", "|---|---:|---:|---:|"]
    for key, metric in result["metrics"].items():
        if metric is not None:
            kappa = "undefined" if metric["cohen_kappa"] is None else f"{metric['cohen_kappa']:.4f}"
            lines.append(f"| {key} | {metric['label_count']} | {metric['agreement']:.4f} | {kappa} |")
    (RESULTS / "judge_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {key: value for key, value in result.items() if key != "records"}


def summarize(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in records:
        groups[(row["configuration"], row["split"])].append(row)
    summaries = []
    metrics = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10", "faithfulness",
               "answer_relevance", "citation_accuracy", "citation_coverage", "abstained"]
    for (configuration, split), rows in groups.items():
        summary = {"configuration": configuration, "split": split, "questions": len(rows),
                   "valid_answers": sum(row.get("answer") is not None for row in rows),
                   "valid_judgements": sum(row.get("judgement") is not None for row in rows),
                   "failed_queries": sum(row.get("error") is not None for row in rows),
                   "unrepresentable_gold": sum(row.get("gold_chunk_count") == 0 for row in rows)}
        for metric in metrics:
            values = [row["metrics"][metric] for row in rows if row.get("metrics", {}).get(metric) is not None]
            summary[metric] = sum(values) / len(values) if values else None
            summary[metric + "_samples"] = len(values)
        for stage in ("retrieval", "answer"):
            timings = [row[stage + "_latency_ms"] for row in rows if row.get(stage + "_latency_ms") is not None]
            for q in (50, 95):
                summary[f"{stage}_p{q}_ms"] = percentile(timings, q / 100) if timings else None
        summary["provider_cost_usd_per_query"] = sum(float(row["cost_usd"]) for row in rows) / len(rows)
        summaries.append(summary)
    return summaries


def write_report(report: dict, stem: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    target = RESULTS / f"{stem}.json"
    temporary = target.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    columns = ["configuration", "split", "questions", "recall@5", "mrr", "ndcg@10", "faithfulness",
               "answer_relevance", "citation_accuracy", "abstained", "retrieval_p50_ms", "retrieval_p95_ms",
               "answer_p50_ms", "answer_p95_ms", "provider_cost_usd_per_query", "valid_judgements", "failed_queries"]
    lines = [f"Status: **{report['status']}**. Judge calibration: **{report['calibration']['status']}**.\n",
             "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for summary in report["summaries"]:
        lines.append("| " + " | ".join("pending" if summary[column] is None else
                                      f"{summary[column]:.4f}" if isinstance(summary[column], float) else str(summary[column])
                                      for column in columns) + " |")
    (RESULTS / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(args: argparse.Namespace) -> None:
    # Check the expensive held-out run's human gate before loading models/calling an API.
    cases = read_jsonl(DATA / "dev.jsonl")
    if args.split == "all" and not args.prepare_labels:
        cases.extend(reviewed_test_cases())
    embeddings = LocalEmbeddings()
    stores = await build_indexes(embeddings)
    client = None if args.retrieval_only else LLMClient(
        settings.EVAL_LLM_BASE_URL, settings.EVAL_LLM_MODEL, settings.EVAL_LLM_API_KEY,
        local=settings.EVAL_LLM_LOCAL, budget_usd=str(settings.EVAL_BUDGET_USD),
        input_usd_per_million=str(settings.EVAL_INPUT_USD_PER_MILLION),
        output_usd_per_million=str(settings.EVAL_OUTPUT_USD_PER_MILLION),
    )
    configurations = CONFIGS if not args.retrieval_only else [row for row in CONFIGS if not row[3]]
    retrievers = {name: HybridRetriever(top_k=10, retrieval_mode=mode, embeddings=embeddings,
                                       store=stores[chunking], learned_reranker=mode == "hybrid_rerank")
                  for name, chunking, mode, rewrite in configurations}
    if args.prepare_labels:
        if client is None:
            raise ValueError("Preparing human labels requires actual model answers")
        grouped = defaultdict(list)
        for row in cases:
            grouped[row["article"]].append(row)
        cases = [rows[index] for rows in grouped.values() for index in (0, 3, 6, 9)]
    records = []
    stem = "calibration_inputs" if args.prepare_labels else "retrieval_" + args.split if args.retrieval_only else "ablations_" + args.split
    report = {"status": "running", "dataset": "squad_v1", "split": args.split,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "python": platform.python_version(), "platform": platform.platform(),
              "package_versions": {package: importlib.metadata.version(package)
                                   for package in ("sentence-transformers", "torch", "langchain-text-splitters", "pydantic")},
              "embedding_model": embeddings.model_name, "embedding_revision": embeddings.revision,
              "judge_model": client.model if client else None,
              "test_review_sha256": hashlib.sha256((DATA / "test_review.jsonl").read_bytes()).hexdigest(),
              "ranking_depth": 10, "warm_index": True, "result_cache": False,
              "experiment_design": "one factor at a time; not a complete factorial or global optimum search",
              "cost_scope": "provider token charges, excluding electricity/hardware", "records": records,
              "calibration": {"status": "pending_human_labels", "agreement": None}}
    try:
        for question_index, case in enumerate(cases):
            selected = [configurations[question_index % len(configurations)]] if args.prepare_labels else configurations
            for name, chunking, mode, rewrite in selected:
                row = {"id": name + ":" + case["id"], "configuration": name, "split": case["split"],
                       "question": case["question"], "reference_answers": [answer["text"] for answer in case["answers"]],
                       "answer_origin": "model_generated", "answer": None, "judgement": None, "metrics": {},
                       "error": None, "cost_usd": "0"}
                started, spent = time.perf_counter(), client.spent if client else 0
                try:
                    results, rewritten = await retrieve_case(retrievers[name], case, rewrite, client)
                    row["retrieval_latency_ms"] = (time.perf_counter() - started) * 1000
                    row["rewritten_query"] = rewritten
                    gold = relevant_chunks(case, list(stores[chunking].chunks.values()))
                    row["gold_chunk_count"] = len(gold)
                    row["metrics"] = ranking_metrics([result.chunk_id for result in results], gold)
                    row["contexts"] = [{"chunk_id": result.chunk_id, "text": result.text,
                                        "document_id": result.document_id, "metadata": result.metadata} for result in results[:3]]
                    if client is not None:
                        answer = await generate_answer(client, case["question"], row["contexts"])
                        row["answer"] = answer.model_dump()
                        row["answer_latency_ms"] = (time.perf_counter() - started) * 1000
                        if not args.prepare_labels:
                            judged = await judge_answer(client, case["question"], row["reference_answers"], row["contexts"], answer)
                            row["judgement"] = judged.model_dump()
                            row["metrics"].update(answer_metrics(answer, judged))
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    if isinstance(error, RuntimeError) and "budget exhausted" in str(error):
                        raise
                finally:
                    row["cost_usd"] = str(client.spent - spent) if client else "0"
                    records.append(row)
                    report["summaries"] = summarize(records)
                    report["calls"] = client.calls if client else []
                    report["provider_cost_usd"] = str(client.spent) if client else "0"
                    write_report(report, stem)
                print(f"{name} {case['id']}: {row['error'] or 'recorded'}", flush=True)
        if args.prepare_labels:
            if any(row["answer"] is None for row in records):
                raise ValueError("Cannot create twenty calibration examples until every model answer succeeds")
            human_label_sheet(records, DATA / "labels_todo.jsonl")
        before_calibration = client.spent if client else 0
        report["calibration"] = await calibrate_judge(client) if client and not args.prepare_labels else {
            "status": "pending_human_labels", "agreement": None}
        report["calibration_cost_usd"] = str(client.spent - before_calibration) if client else "0"
        report["provider_cost_usd"] = str(client.spent) if client else "0"
        report["status"] = "partial_failures" if any(row["error"] for row in records) else "measured"
        if not args.prepare_labels and not args.retrieval_only:
            dev = [row for row in report["summaries"] if row["split"] == "dev"
                   and row["recall@5_samples"] == row["questions"]]
            report["selected_configuration"] = max(dev, key=lambda row: (row["recall@5"], row["ndcg@10"],
                                                                         -row["retrieval_p50_ms"]))["configuration"] if dev else None
        write_report(report, stem)
    except Exception as error:
        report["status"] = "aborted"
        report["error"] = f"{type(error).__name__}: {error}"
        report.setdefault("summaries", summarize(records))
        write_report(report, stem)
        raise
    finally:
        if client:
            await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "all"), default="all")
    parser.add_argument("--retrieval-only", action="store_true", help="Debug retrieval only; omits rewriting and answer metrics")
    parser.add_argument("--prepare-labels", action="store_true", help="Generate actual dev answers with blank human labels")
    args = parser.parse_args()
    if args.prepare_labels and args.retrieval_only:
        parser.error("Label preparation requires actual LLM answers")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
