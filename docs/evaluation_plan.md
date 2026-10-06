# EvidenceRAG evaluation checkpoint

Scope: this repository only. Display name: EvidenceRAG. Do not rename the GitHub
repository. Corpus approval: the user approved SQuAD 1.1, CC BY-SA 4.0, with an
article-disjoint development/test subset and a manual review of every test case.

## Baseline before behaviour changes

The original suite passed in an isolated Python environment using the existing
requirements. It covers ingestion, graph routing, API/SSE, local tool examples,
cache/compression, and the synthetic overlap-based evaluation. Those tests do
not prove retrieval quality, factual answer quality, real cross-encoder scoring,
PostgreSQL persistence, token streaming during generation, or stage tracing.

Current flow: upload -> recursive chunks -> hash/OpenAI embeddings -> in-memory
dense and BM25-like search -> RRF -> keyword reranking -> heuristic grading and
query rewrites -> copied-context answer -> delayed SSE output.

Known evaluation hazards to report before fixing: cache state is shared between
retriever instances and keyed only by query text; provider errors silently fall
back to hash vectors; BM25 document frequency uses substring matching; a reranker
constructed with a fixed output count can truncate a larger requested top-k.

These hazards were reproduced by failing regression checks before fixes. The
shared fixes isolate each retriever's cache, include result options in cache
keys, preserve the requested reranker output count, use exact-token BM25 document
frequency and measured mean chunk length, and propagate embedding-provider errors.
The original suite and new dataset/regression checks passed after these changes.
GitHub Actions is configured; no hosted workflow run is claimed before pushing.

## Required gates

- Report failing tests before changing the affected behaviour. Preserve the
  baseline comparison; record every behaviour fix and its regression check.
- Golden questions remain original human annotations. Disclose any generated
  supplemental data as synthetic. The user reviews and corrects the test split.
- Create `labels_todo.jsonl` from actual model answers and retrieved/cited evidence,
  with a labelling guide. The user supplies all judge-validation labels; never
  pre-fill them or claim agreement while labels are missing.
- Use a local, small instruction model for generation and judging by default.
  Paid providers must enforce a total full-run spend strictly below $3, including
  rewrites, answer generation, judging, calibration, and retries. Reserve request
  cost before dispatch; record model identity, token usage, and costs. Missing
  usage or prices must not become a zero-cost measurement.
- Add a provider interface, genuine local embedding and cross-encoder models,
  source-aware chunking, ranking metrics, answer/citation judging, and ablations.
  Use dev results to select the configuration; do not tune on held-out answers.
- `make eval` generates JSON records and Markdown rows for every requested
  configuration, including measured latency and token cost. Missing measurements
  remain missing, not invented values.
- Stop after the ablations. Show the results table, explain every metric in
  plain language, and explain the winning configuration using measured evidence.
  Wait for the user's approval before pgvector, corrective RAG, SSE, or tracing
  implementation. Human-review and label completion are separate prerequisites
  for accepting held-out results and judge agreement.
- After checkpoint approval, implement and verify production features, then
  attempt the optional embedding fine-tuning using separate training data.
- For each interview-note decision, record choice, alternative, reason, and
  measured evidence. The user writes the final resume bullets from the results
  table; no fabricated performance claims or premature resume edits.
