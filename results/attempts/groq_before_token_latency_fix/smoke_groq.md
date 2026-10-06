Status: **measured**. Judge calibration: **pending_human_labels**.

| configuration | split | questions | recall@5 | mrr | ndcg@10 | faithfulness | answer_relevance | citation_accuracy | abstained | retrieval_p50_ms | retrieval_p95_ms | answer_p50_ms | answer_p95_ms | provider_cost_usd_per_query | valid_judgements | failed_queries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_hybrid | dev | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 186.6873 | 186.6873 | 2105.8961 | 2105.8961 | 0.00070965 | 1 | 0 |
| recursive_bm25 | dev | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 20.1776 | 20.1776 | 11947.3203 | 11947.3203 | 0.00054690 | 1 | 0 |
| recursive_hybrid_rerank | dev | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1223.2812 | 1223.2812 | 11909.2159 | 11909.2159 | 0.00054885 | 1 | 0 |
| semantic_hybrid | dev | 1 | 0.5000 | 1.0000 | 0.8175 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 181.5565 | 181.5565 | 11832.9103 | 11832.9103 | 0.00048150 | 1 | 0 |
| recursive_hybrid_rewrite | dev | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 11827.2089 | 11827.2089 | 35860.8723 | 35860.8723 | 0.00060135 | 1 | 0 |
