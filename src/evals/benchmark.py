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
from urllib.parse import urlparse
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field
import httpx
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
from src.rag.reranker import CrossEncoderReranker

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "squad_v1"
RESULTS = ROOT / "results"
# User-requested sensitivity analysis: zero-based review row 23 (file line 24).
AMBIGUOUS_TEST_ID = "573786b51c4567190057448e"
GARBLED_FORMULA_ID = "57297a276aef051400154f8a"
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


def restore_checkpoint(path: Path, identity: dict, resume: bool) -> dict | None:
    if not path.exists():
        if resume:
            raise ValueError("No checkpoint exists to resume")
        return None
    if not resume:
        raise ValueError("Results already exist; use --resume or archive them before a new run")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("run_identity") != identity:
        raise ValueError("Checkpoint model, prices, inputs, or code changed; archive it before a new run")
    if len({row["id"] for row in report["records"]}) != len(report["records"]):
        raise ValueError("Checkpoint contains duplicate case IDs")
    return report


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
        row["reviewer"] = review["reviewer"]
        row["review_notes"] = review.get("notes")
        row["review_provenance"] = "ai_assisted" if "ai-assisted" in review["reviewer"].lower() else "reviewer_declared"
        if review["question"] != row["question"] or review["source_chunk_id"] != row["source_chunk_id"]:
            raise ValueError("Question/source changes require a new version, not a silent review-file edit")
        if review.get("corrected_answers") is not None:
            answers = review["corrected_answers"]
            if not isinstance(answers, list) or not answers:
                raise ValueError("Corrected answers must be a nonempty list of text/answer_start records")
            row["answers"] = answers
            row["origin"] = "review_corrected_squad"
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
        if row["split"] == "test" and row["question_id"] != AMBIGUOUS_TEST_ID:
            groups[(row["configuration"], "test_without_ambiguous")].append(row)
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
    # ponytail: rewrite the bounded run snapshot after each case; use JSONL
    # append plus periodic summaries if benchmark size makes this I/O material.
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
    if report.get("test_review_provenance"):
        lines.insert(1, "Test review: " + ", ".join(report["test_review_provenance"]) +
                     ". AI-assisted review remains provisional pending the user's spot-check.\n")
        lines.insert(2, f"`test` includes all 40 questions; `test_without_ambiguous` excludes `{AMBIGUOUS_TEST_ID}` "
                     "(zero-based row 23 / file line 24). Both use the same frozen predictions. "
                     f"Formula case `{GARBLED_FORMULA_ID}` retains the source's garbled formula; "
                     "exact-match scoring is not computed.\n")
    for summary in report["summaries"]:
        lines.append("| " + " | ".join("pending" if summary[column] is None else
                                      format(summary[column], ".8f" if "cost" in column else ".4f")
                                      if isinstance(summary[column], float) else str(summary[column])
                                      for column in columns) + " |")
    (RESULTS / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(args: argparse.Namespace) -> None:
    # Check the expensive held-out run's human gate before loading models/calling an API.
    cases = read_jsonl(DATA / "dev.jsonl")
    smoke = getattr(args, "smoke", False)
    resume = getattr(args, "resume", False)
    hostname = urlparse(settings.EVAL_LLM_BASE_URL).hostname
    provider = "groq" if hostname == "api.groq.com" else "gemini" if hostname == "generativelanguage.googleapis.com" else "local"
    if args.split == "all" and not args.prepare_labels and not smoke:
        cases.extend(reviewed_test_cases())
    if smoke:
        # One dev question per article; exercise generation, judging, and rewriting.
        cases = [next(row for row in cases if row["article"] == article)
                 for article in dict.fromkeys(row["article"] for row in cases)]
    stem = "smoke_" + provider if smoke else "calibration_inputs" if args.prepare_labels else "retrieval_" + args.split if args.retrieval_only else "ablations_" + provider + "_" + args.split
    identity = {"model": settings.EVAL_LLM_MODEL, "base_url": settings.EVAL_LLM_BASE_URL,
                "rates": [str(settings.EVAL_INPUT_USD_PER_MILLION), str(settings.EVAL_OUTPUT_USD_PER_MILLION)],
                "cases_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
                "corpus_sha256": hashlib.sha256((DATA / "corpus.jsonl").read_bytes()).hexdigest(),
                "code_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in sorted(
                    list((ROOT / "src" / "rag").glob("*.py")) + list((ROOT / "src" / "evals").glob("*.py")) + [ROOT / "src" / "llm.py"]))).hexdigest()}
    previous = restore_checkpoint(RESULTS / (stem + ".json"), identity, resume)
    if previous and previous["status"] == "measured":
        print("Checkpoint already complete; no calls repeated", flush=True)
        return
    embeddings = LocalEmbeddings()
    stores = await build_indexes(embeddings)
    client = None if args.retrieval_only else LLMClient(
        settings.EVAL_LLM_BASE_URL, settings.EVAL_LLM_MODEL, settings.EVAL_LLM_API_KEY,
        local=settings.EVAL_LLM_LOCAL, budget_usd=str(settings.EVAL_BUDGET_USD),
        input_usd_per_million=str(settings.EVAL_INPUT_USD_PER_MILLION),
        output_usd_per_million=str(settings.EVAL_OUTPUT_USD_PER_MILLION),
        request_interval_seconds=0 if settings.EVAL_LLM_LOCAL else settings.EVAL_REQUEST_INTERVAL_SECONDS,
        max_rate_limit_retries=settings.EVAL_RATE_LIMIT_RETRIES,
        tokens_per_minute=settings.EVAL_TOKENS_PER_MINUTE if provider == "groq" else 0,
        tokens_per_day=settings.EVAL_TOKENS_PER_DAY if provider == "groq" else 0,
        quota_path=ROOT / ".cache" / ("groq_quota_" + hashlib.sha256((settings.EVAL_LLM_API_KEY + settings.EVAL_LLM_MODEL).encode()).hexdigest()[:16] + ".json") if provider == "groq" else None,
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
    records = previous["records"] if previous else []
    report = {"status": "running", "dataset": "squad_v1", "split": args.split,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "python": platform.python_version(), "platform": platform.platform(),
              "package_versions": {package: importlib.metadata.version(package)
                                   for package in ("sentence-transformers", "torch", "langchain-text-splitters", "pydantic")},
              "embedding_model": embeddings.model_name, "embedding_revision": embeddings.revision,
              "reranker_model": CrossEncoderReranker.model_name, "reranker_revision": CrossEncoderReranker.revision,
              "chunk_indexes": {mode: {"chunks": len(store.chunks)} for mode, store in stores.items()},
              "judge_model": client.model if client else None,
              "same_model_judge": client is not None,
              "run_identity": identity, "provider": provider,
              "token_limits": {"minute": settings.EVAL_TOKENS_PER_MINUTE, "day": settings.EVAL_TOKENS_PER_DAY} if provider == "groq" else None,
              "request_interval_seconds": client.request_interval_seconds if client else 0,
              "rate_limit_retries": client.max_rate_limit_retries if client else 0,
              "cost_rates_usd_per_million": {"input": str(client.input_price), "output": str(client.output_price)} if client else None,
              "test_review_sha256": hashlib.sha256((DATA / "test_review.jsonl").read_bytes()).hexdigest(),
              "test_review_provenance": sorted({case["review_provenance"] for case in cases if case["split"] == "test"}),
              "test_sensitivity_excluded_id": AMBIGUOUS_TEST_ID,
              "garbled_formula_id": GARBLED_FORMULA_ID, "exact_match_computed": False,
              "ranking_depth": 10, "warm_index": True, "result_cache": False,
              "experiment_design": "one factor at a time; not a complete factorial or global optimum search",
              "cost_scope": "provider token charges, excluding electricity/hardware", "records": records,
              "calibration": {"status": "pending_human_labels", "agreement": None}}
    if previous:
        report = previous
        report["status"] = "running"
        if client:
            client.calls = report.get("calls", [])
            client.spent = Decimal(report.get("provider_cost_usd", "0"))
    def checkpoint():
        if not row.get("completed"):
            row["cost_usd"] = str(prior_cost + client.spent - spent)
            row.setdefault("call_range", [call_start, call_start])[1] = len(client.calls)
        report["calls"] = client.calls
        report["provider_cost_usd"] = str(client.spent)
        report["waiting_until_unix"] = client.wait_until
        report["summaries"] = summarize(records)
        write_report(report, stem)
    if client:
        client.checkpoint = checkpoint
    try:
        for question_index, case in enumerate(cases):
            selected = [configurations[(0, 2, 4, 5, 6)[question_index]]] if smoke else [configurations[question_index % len(configurations)]] if args.prepare_labels else configurations
            for name, chunking, mode, rewrite in selected:
                existing = next((row for row in records if row["id"] == name + ":" + case["id"]), None)
                if existing and existing.get("completed"):
                    continue
                row = {"id": name + ":" + case["id"], "configuration": name, "split": case["split"],
                       "question_id": case["id"], "question_origin": case["origin"],
                       "reviewer": case.get("reviewer"), "review_provenance": case.get("review_provenance"),
                       "review_notes": case.get("review_notes"),
                       "gold_source_chunk_id": case["source_chunk_id"],
                       "question": case["question"], "reference_answers": [answer["text"] for answer in case["answers"]],
                       "answer_origin": "model_generated", "answer": None, "judgement": None, "metrics": {},
                       "error": None, "cost_usd": "0"}
                if existing:
                    row = existing
                    row["error"] = None
                else:
                    records.append(row)
                row["completed"] = False
                prior_cost = Decimal(row["cost_usd"])
                started, spent = time.perf_counter(), client.spent if client else 0
                call_start = len(client.calls) if client else 0
                try:
                    if "contexts" not in row:
                        results, rewritten = await retrieve_case(retrievers[name], case, rewrite, client)
                        row["retrieval_latency_ms"] = (time.perf_counter() - started) * 1000
                        row["rewritten_query"] = rewritten
                        gold = relevant_chunks(case, list(stores[chunking].chunks.values()))
                        row["gold_chunk_count"] = len(gold)
                        row["gold_chunk_ids"] = sorted(gold)
                        row["ranked_chunk_ids"] = [result.chunk_id for result in results]
                        row["metrics"] = ranking_metrics([result.chunk_id for result in results], gold)
                        row["contexts"] = [{"chunk_id": result.chunk_id, "text": result.text,
                                            "document_id": result.document_id, "metadata": result.metadata} for result in results[:3]]
                    if client is not None:
                        if row["answer"] is None:
                            answer = await generate_answer(client, case["question"], row["contexts"])
                            row["answer"] = answer.model_dump()
                            row["answer_latency_ms"] = (time.perf_counter() - started) * 1000
                            checkpoint()
                        else:
                            answer = Answer.model_validate(row["answer"])
                        if not args.prepare_labels:
                            judged = await judge_answer(client, case["question"], row["reference_answers"], row["contexts"], answer)
                            row["judgement"] = judged.model_dump()
                            row["metrics"].update(answer_metrics(answer, judged))
                    row["completed"] = True
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    if isinstance(error, RuntimeError) and any(message in str(error).lower() for message in ("budget exhausted", "daily gemini quota exhausted")):
                        raise
                    if getattr(error, "response", None) is not None and error.response.status_code in (401, 403, 404, 429):
                        raise
                    if isinstance(error, (httpx.TransportError, httpx.HTTPStatusError)):
                        raise
                    row["completed"] = True
                finally:
                    row["cost_usd"] = str(prior_cost + client.spent - spent) if client else "0"
                    row.setdefault("call_range", [call_start, call_start])[1] = len(client.calls) if client else 0
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
        report["calibration"] = await calibrate_judge(client) if client and not args.prepare_labels and not smoke else {
            "status": "pending_human_labels", "agreement": None}
        report["calibration_cost_usd"] = str(client.spent - before_calibration) if client else "0"
        report["provider_cost_usd"] = str(client.spent) if client else "0"
        report["status"] = "partial_failures" if any(row["error"] for row in records) else "measured"
        if not args.prepare_labels and not args.retrieval_only and not smoke:
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
    parser.add_argument("--smoke", action="store_true", help="Five dev cases across articles; includes a rewrite call")
    parser.add_argument("--resume", action="store_true", help="Resume a matching checkpoint without repeating completed cases")
    args = parser.parse_args()
    if args.prepare_labels:
        args.split = "dev"
    if args.smoke:
        args.split = "dev"
        if args.prepare_labels or args.retrieval_only:
            parser.error("Smoke mode requires answering and judging, without label preparation")
    if args.prepare_labels and args.retrieval_only:
        parser.error("Label preparation requires actual LLM answers")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
