Status: **running**. Judge calibration: **pending_human_labels**.

Test review: ai_assisted. AI-assisted review remains provisional pending the user's spot-check.

`test` includes all 40 questions; `test_without_ambiguous` excludes `573786b51c4567190057448e` (zero-based row 23 / file line 24). Both use the same frozen predictions. Formula case `57297a276aef051400154f8a` retains the source's garbled formula; exact-match scoring is not computed.

Token/cost projection: {"sample_questions": 61, "remaining_configuration_questions": 639, "remaining_usd": "2.74214310", "limitations": "Projection from completed samples, including observed retry reservations; article/configuration mix, external account use and SambaNova limits can change completion time/cost. Quota-days are a Groq lower bound, not an ETA.", "groq_tokens_per_question": 953.983606557377, "projected_remaining_groq_tokens": 609595.524590164, "groq_quota_days": 3.0479776229508198, "projected_remaining_cost_usd": "0.7474760114754098360655737704", "budget_sufficient": true, "affordable_questions": 2344}

Call diagnostics: {"calls": 131, "finish_reason_recorded": 131, "finish_reason_length": 0, "truncations": 0, "http_400s": 0, "http_429s": 0}

| configuration | split | questions | recall@5 | mrr | ndcg@10 | faithfulness | answer_relevance | citation_accuracy | abstained | retrieval_p50_ms | retrieval_p95_ms | answer_provider_p50_ms | answer_provider_p95_ms | judge_provider_p50_ms | judge_provider_p95_ms | answer_queue_p50_ms | answer_queue_p95_ms | answer_wall_p50_ms | answer_wall_p95_ms | provider_cost_usd_per_query | valid_judgements | failed_queries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_hybrid | dev | 9 | 1.0000 | 0.8148 | 0.8624 | 1.0000 | 0.8889 | 1.0000 | 0.1111 | 107.9751 | 167.5195 | 723.2949 | 1076.0272 | 473.1734 | 694.4512 | 11.9250 | 22.5460 | 874.6836 | 1200.3646 | 0.00119606 | 9 | 0 |
| recursive_dense | dev | 9 | 0.7778 | 0.6486 | 0.7294 | 1.0000 | 0.7778 | 1.0000 | 0.2222 | 61.5988 | 66.3156 | 853.0050 | 1110.2693 | 402.2355 | 482.9921 | 12.8651 | 10434.0067 | 962.1522 | 11260.7324 | 0.00111339 | 9 | 0 |
| recursive_bm25 | dev | 9 | 0.8889 | 0.9012 | 0.9223 | 1.0000 | 0.8889 | 1.0000 | 0.0000 | 34.5357 | 58.4445 | 791.8694 | 1205.9028 | 396.7914 | 594.7001 | 11.7239 | 21.9610 | 882.8115 | 1247.0925 | 0.00116724 | 9 | 0 |
| recursive_hybrid | dev | 9 | 1.0000 | 0.7963 | 0.8479 | 1.0000 | 0.7778 | 1.0000 | 0.1111 | 100.8313 | 127.4012 | 835.7463 | 1103.1038 | 441.5563 | 699.8392 | 13.3342 | 20.2289 | 973.4826 | 1227.2640 | 0.00118461 | 9 | 0 |
| recursive_hybrid_rerank | dev | 9 | 1.0000 | 0.9259 | 0.9444 | 1.0000 | 0.7778 | 1.0000 | 0.0000 | 1295.7372 | 2491.3882 | 928.4980 | 993.4358 | 458.1345 | 593.3059 | 11.9055 | 22.4110 | 2115.0993 | 3437.2033 | 0.00119424 | 9 | 0 |
| semantic_hybrid | dev | 9 | 1.0000 | 0.8426 | 0.8776 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 173.7432 | 228.7883 | 781.5244 | 1161.7988 | 381.3684 | 626.2852 | 13.8602 | 21.4786 | 985.2153 | 1365.9305 | 0.00101611 | 8 | 0 |
| recursive_hybrid_rewrite | dev | 8 | 1.0000 | 0.8542 | 0.8914 | 1.0000 | 0.7500 | 1.0000 | 0.1250 | 864.7134 | 971.8171 | 816.8451 | 1048.4851 | 431.8370 | 463.6992 | 11239.3346 | 11362.9097 | 12946.6459 | 13168.1581 | 0.00122091 | 8 | 0 |
