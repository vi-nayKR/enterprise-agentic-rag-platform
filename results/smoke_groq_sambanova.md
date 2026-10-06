Status: **blocked by SambaNova HTTP 402 PAYMENT_METHOD_REQUIRED**.

The five-case cross-family smoke has not passed; the full run has not started.
Configuration: Groq `openai/gpt-oss-120b` for answering/rewriting (low reasoning,
1024-token ceiling), SambaNova `DeepSeek-V3.1` for judging (1024-token ceiling,
30-second pacing), same configured accounts/keys, one combined spending cap.

| Observed measurement | Value |
| --- | ---: |
| Saved valid Groq answers | 1 of 5 |
| Valid SambaNova judgements | 0 of 5 |
| Groq prompt / completion tokens | 807 / 193 |
| Unknown-usage judge attempts | 2 |
| Truncations / HTTP 400s / HTTP 429s | 0 / 0 / 0 |
| Groq provider latency | 1027.59 ms |
| First judge attempt | Read timeout after 180 s |
| Cached-answer judge retry | HTTP 402, 886.35 ms |
| Combined measured/reserved smoke allowance | $0.05860485 |
| Remaining after all attempts and this smoke | $2.75515035 |

No provider charges are inferred from a failed request. Unknown usage retains a
conservative worst-case reservation so retries cannot silently exceed the cap.
Raw redacted responses and the cached checkpoint remain local until this run
can finish; no actual account billing details are published here.

The single measured answer used 1000 tokens. At that rate, 700 answers would
use 700K Groq tokens, or **3.5 full 200K-token quota-days**, plus rewrites and
quota/service waiting. This is a one-question, answer-only projection, not an ETA.
Judge usage/cost projections and the final five-case latency table remain pending.

Daily-quota 429s automatically wait for reset using the same key. HTTP 402 is a
billing-access failure, not a resettable token quota; it stops without retries.
After the user restores existing-account access, resume the saved answer with
`python -m src.evals.benchmark --smoke --resume`. A clean five-case smoke still
gates the fresh full run. Human judge-validation labels remain pending.
