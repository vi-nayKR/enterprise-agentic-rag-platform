Status: **interrupted**. Judge calibration: **pending_human_labels**.

Test review: ai_assisted. AI-assisted review remains provisional pending the user's spot-check.

`test` includes all 40 questions; `test_without_ambiguous` excludes `573786b51c4567190057448e` (zero-based row 23 / file line 24). Both use the same frozen predictions. Formula case `57297a276aef051400154f8a` retains the source's garbled formula; exact-match scoring is not computed.

| configuration | split | questions | recall@5 | mrr | ndcg@10 | faithfulness | answer_relevance | citation_accuracy | abstained | retrieval_p50_ms | retrieval_p95_ms | answer_p50_ms | answer_p95_ms | provider_cost_usd_per_query | valid_judgements | failed_queries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_hybrid | dev | 1 | 1.0000 | 1.0000 | 1.0000 | pending | pending | pending | pending | 241.1611 | 241.1611 | 35230.8379 | 35230.8379 | 0.00000000 | 0 | 1 |
| recursive_dense | dev | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 664.4753 | 664.4753 | 69340.4583 | 69340.4583 | 0.00000000 | 1 | 0 |
| recursive_bm25 | dev | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 101.1292 | 101.1292 | 33827.3501 | 33827.3501 | 0.00000000 | 1 | 0 |
| recursive_hybrid | dev | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 553.6029 | 553.6029 | 30531.8581 | 30531.8581 | 0.00000000 | 1 | 0 |
| recursive_hybrid_rerank | dev | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 4321.9298 | 4321.9298 | 57906.1333 | 57906.1333 | 0.00000000 | 1 | 0 |
