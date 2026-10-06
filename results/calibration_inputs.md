Status: **measured**. Judge calibration: **pending_human_labels**.

| configuration | split | questions | recall@5 | mrr | ndcg@10 | faithfulness | answer_relevance | citation_accuracy | abstained | retrieval_p50_ms | retrieval_p95_ms | answer_p50_ms | answer_p95_ms | provider_cost_usd_per_query | valid_judgements | failed_queries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_hybrid | dev | 3 | 1.0000 | 1.0000 | 1.0000 | pending | pending | pending | pending | 193.3728 | 576.5320 | 59881.0042 | 68067.1197 | 0.0000 | 0 | 0 |
| recursive_dense | dev | 3 | 1.0000 | 0.7778 | 0.8333 | pending | pending | pending | pending | 99.0229 | 324.0996 | 44002.5424 | 47043.4926 | 0.0000 | 0 | 0 |
| recursive_bm25 | dev | 3 | 1.0000 | 1.0000 | 1.0000 | pending | pending | pending | pending | 73.7739 | 138.5517 | 48711.5234 | 56268.3426 | 0.0000 | 0 | 0 |
| recursive_hybrid | dev | 3 | 0.8333 | 0.7778 | 0.7044 | pending | pending | pending | pending | 215.2320 | 455.4874 | 61705.7237 | 63142.8273 | 0.0000 | 0 | 0 |
| recursive_hybrid_rerank | dev | 3 | 1.0000 | 1.0000 | 1.0000 | pending | pending | pending | pending | 2121.0571 | 3475.5048 | 58347.7018 | 60762.7930 | 0.0000 | 0 | 0 |
| semantic_hybrid | dev | 3 | 0.8333 | 1.0000 | 0.9392 | pending | pending | pending | pending | 386.5822 | 387.1301 | 39934.6806 | 53471.8329 | 0.0000 | 0 | 0 |
| recursive_hybrid_rewrite | dev | 2 | 1.0000 | 0.7500 | 0.8155 | pending | pending | pending | pending | 2481.2526 | 4658.2949 | 54138.1572 | 86418.2657 | 0.0000 | 0 | 0 |
