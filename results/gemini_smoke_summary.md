# Gemini smoke outcome

Status: **blocked by daily free-tier quota**. Full Gemini run: **not started**.

| Item | Observed result |
|---|---|
| Cases | 5 requested; 3 finalized; case 4 interrupted during backoff; case 5 unstarted |
| Valid answers / judgements | 0 / 0 |
| Persisted attempts including diagnostics | 11; {404: 2, 503: 3, 429: 6} |
| Token usage | Unavailable for rejected requests; no measured-zero token claim |
| Persisted cost reservations | $0.0728912 |
| Unflushed attempts, conservative additional reservation | At most 7 attempts; $0.056448 |
| Combined conservative reservation upper bound | $0.1293392; not billed charges |
| Pacing / backoff | 12-second minimum interval; exponential 2/4/8/16/32/60 seconds |
| Quota | 20 free-tier requests per model per day; observed reset delay 59,707 seconds (~16.6 hours) |

Gemini 2.5 Flash was refused for new users (404). The provider directed migration to 3.8 Flash.
3.8 returned 503 failures and then 429 daily-quota exhaustion. No quality scores or winner can be inferred.
The updated client stops on explicit daily-quota errors; short-lived 429 and 5xx errors retain metered backoff.
The full benchmark requires about 1,500 calls. It remains pending user choice: paid access, quota reset, or local inference.
Same-model judge remains a limitation, and independent human calibration labels remain pending.
