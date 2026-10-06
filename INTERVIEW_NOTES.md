# EvidenceRAG: decisions and evidence

| Decision | What we chose | Alternative | Why | Measured evidence |
|---|---|---|---|---|
| Benchmark source | Original human-authored SQuAD questions, with article-disjoint dev/test splits | LLM-generated question/answer pairs | Public source spans are independently traceable; synthetic gold could reward generation biases | Dataset inventory in `results/dataset_v1.json`; retrieval/answer outcomes pending |
| Review ownership | User reviews test cases and labels frozen judge examples | Agent supplies evaluation labels | Independence is necessary for defensible judge agreement | Human review and judge agreement remain pending; no proxy agreement claimed |
| Embeddings | Pinned local MiniLM sentence embeddings | Hash vectors or paid API embeddings | Real learned similarity without provider charges; no silent fallback between vector spaces | Model loads successfully; comparative retrieval measurements pending |
| Reranking | Real pinned cross-encoder as an explicit ablation | Existing keyword-coverage reranker | Test whether learned pairwise ranking pays for its latency | Comparative recall and latency pending |
| Chunking | Fixed, existing-library recursive, and adjacent-sentence embedding boundaries | Handwritten recursive splitter alone | The old splitter failed a long-unbroken-input regression; offsets are needed for citation evidence | Source-offset/coverage regression checks pass; comparative chunking outcomes pending |
| Cache isolation | Per-retriever caches with request-option keys; caches off during measurement | Global query-only cache | Separate stores and configurations must not contaminate each other's results | Regression reproduced cross-store leakage and requested-result truncation before fixes; checks now pass |
| Lexical retrieval | Token-level BM25 document frequency and actual average chunk length | Substring frequency and a fixed length constant | Ranking must implement the stated lexical scoring method | Analytic BM25 regression matches the expected score |
| Inference and budget | One local compatible client with request ledger and conservative paid-call reservation | Separate unmetered clients for generation and judging | One spending cap must cover every stage and failure | Local smoke request logged token usage with no provider charge; full-run costs pending |
| Production storage | Keep the current memory store until ablation checkpoint approval | Implement pgvector immediately | Establish retrieval evidence before adding operational complexity | Persistence is not implemented or claimed |

## What failed and what remains uncertain

The original tests passed but did not catch shared-cache leakage, result-count
truncation, substring-based BM25 frequency, silent embedding fallback, or unbounded
chunks for unbroken text. Added checks reproduced these failures before shared
fixes. This is why a passing demo suite did not establish measured RAG quality.

The benchmark is public single-hop Wikipedia data and may have been present in
model pretraining. It does not establish unseen-corpus generalisation or enterprise
readiness. The semantic boundary heuristic is deliberately simple and must earn
its place through the ablation results. A small local judge may disagree with
human labels; report that outcome rather than changing labels or hiding cases.

Performance, operational features, and final resume claims remain pending.
Populate the measured-evidence column from committed results at the ablation
checkpoint; no winning configuration is asserted before measurements.
