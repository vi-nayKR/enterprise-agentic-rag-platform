# Groq five-question smoke

Status: **passed**. Same-model judge; independent calibration pending.

| Item | Result |
|---|---|
| Questions / valid answers / valid judgements | 5 / 5 / 5 |
| Failures | 0 |
| Calls including one rewrite | 11 |
| Input / output tokens | 11035 / 2055 |
| Paid-rate estimate | $0.00288825 |
| Rate-limit responses / retries | 0 / 0 |
| Pacing | 12-second minimum request interval plus conservative 8K TPM / 200K TPD rolling quotas |
| Observed waits | Local token-quota waits; no provider rate-limit error |
| Resume verification | Complete checkpoint returned without additional calls |
| Remaining full-run cap | $2.85777255, including allowance for prior Gemini attempts |

Cost is a paid-rate equivalent, not a billing statement. No live retry success is claimed because no 429 occurred.
Tests exercise backoff, failed-call reservations, daily windows, persisted quota usage, and recovery after interrupted judging.
Five smoke examples validate the execution path; they are not the full ablation benchmark or evidence of a winning configuration.
Raw rankings, answers, judge predictions, tokens and provider rate-limit headers are in `smoke_groq.json`.
