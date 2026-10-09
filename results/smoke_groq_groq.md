Status: **measured**. Judge calibration: **pending_human_labels**.

Token/cost projection: {"sample_questions": 5, "remaining_configuration_questions": 0, "remaining_usd": "2.80720675", "limitations": "Projection from completed samples, including observed retry reservations; article/configuration mix, external account use and SambaNova limits can change completion time/cost. Quota-days are a Groq lower bound, not an ETA.", "groq_tokens_per_question": 977.4, "projected_remaining_groq_tokens": 0.0, "groq_quota_days": 0.0, "projected_remaining_cost_usd": "0E-8", "budget_sufficient": true, "affordable_questions": 2143}

Call diagnostics: {"calls": 10, "finish_reason_recorded": 10, "finish_reason_length": 0, "truncations": 0, "http_400s": 0, "http_429s": 0}

| configuration | split | questions | recall@5 | mrr | ndcg@10 | faithfulness | answer_relevance | citation_accuracy | abstained | retrieval_p50_ms | retrieval_p95_ms | answer_provider_p50_ms | answer_provider_p95_ms | judge_provider_p50_ms | judge_provider_p95_ms | answer_queue_p50_ms | answer_queue_p95_ms | answer_wall_p50_ms | answer_wall_p95_ms | provider_cost_usd_per_query | valid_judgements | failed_queries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_hybrid | dev | 5 | 1.0000 | 0.9000 | 0.9262 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 94.9741 | 171.3817 | 869.8153 | 1281.0736 | 526.5479 | 691.7890 | 4.8644 | 10495.9367 | 1050.3381 | 11401.4871 | 0.00130969 | 5 | 0 |
