Status: **running**. Judge calibration: **pending_human_labels**.

Test review: ai_assisted. AI-assisted review remains provisional pending the user's spot-check.

`test` includes all 40 questions; `test_without_ambiguous` excludes `573786b51c4567190057448e` (zero-based row 23 / file line 24). Both use the same frozen predictions. Formula case `57297a276aef051400154f8a` retains the source's garbled formula; exact-match scoring is not computed.

| configuration | split | questions | recall@5 | mrr | ndcg@10 | faithfulness | answer_relevance | citation_accuracy | abstained | retrieval_p50_ms | retrieval_p95_ms | answer_p50_ms | answer_p95_ms | provider_cost_usd_per_query | valid_judgements | failed_queries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_hybrid | dev | 6 | 1.0000 | 0.8333 | 0.8770 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 144.0563 | 327.9741 | 11581.0805 | 12285.1025 | 0.00108227 | 4 | 2 |
| recursive_dense | dev | 6 | 0.6667 | 0.5840 | 0.6775 | 1.0000 | 0.4000 | 1.0000 | 0.6000 | 67.0882 | 283.2741 | 11565.5218 | 11760.1621 | 0.00073127 | 5 | 1 |
| recursive_bm25 | dev | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 49.3704 | 92.7673 | 11504.7818 | 56082.4141 | 0.00058101 | 5 | 0 |
| recursive_hybrid | dev | 5 | 1.0000 | 0.9000 | 0.9262 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 132.8661 | 575.8174 | 11947.9838 | 12191.0826 | 0.00059571 | 5 | 0 |
| recursive_hybrid_rerank | dev | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8000 | 0.0000 | 1835.5046 | 2203.3727 | 11359.5229 | 12072.1668 | 0.00058749 | 5 | 0 |
| semantic_hybrid | dev | 5 | 1.0000 | 0.8667 | 0.8935 | 1.0000 | 1.0000 | 0.8000 | 0.0000 | 217.9437 | 426.8723 | 11732.1733 | 12103.8799 | 0.00053838 | 5 | 0 |
| recursive_hybrid_rewrite | dev | 5 | 1.0000 | 0.9000 | 0.9262 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 11594.7363 | 11745.1161 | 35307.4263 | 36075.5648 | 0.00067794 | 5 | 0 |
