# Stopped Groq attempt

Stopped at the user's request on 2026-10-06. The six original result files in
this directory are preserved unchanged; their recorded running status is historical.
The replacement run must start fresh rather than resume these predictions.

- 37 configuration/question records finalized; 80 HTTP attempts.
- Estimated provider cost, including unknown-usage reservations: $0.02578395.
- Three HTTP 400 attempts affected two question/configuration records. All three
  provider messages explicitly say the completion-token ceiling prevented a valid
  JSON document; the required `reason` property was missing from truncated output.
  This is output exhaustion, not evidence of a context-window or unsupported-schema problem.
- A third failed record was a judge citation-label validation failure; it remains
  visible in the original output and was not relabelled as successful.
- Exact historical `finish_reason="length"` count is **unknown**: the old client
  retained no finish reasons across the 80 attempts. At least three structured
  outputs were truncated, as confirmed by those HTTP 400 messages.
- The old client retained error messages, but not full response bodies. New calls
  retain error bodies with the configured API key redacted.
- Old answer/judge latency contains throttle waiting and is not provider latency.

Replacement limits are configurable: answer 1024, judge 1024, rewrite 256 tokens.
Strict JSON schema validation and truncation-as-failure remain enabled. GPT-OSS
uses Groq's explicit `max_completion_tokens` parameter, with low reasoning effort.
New logs count truncations/400s and separate provider, queue, and answer wall time.
The smoke uses the first five dev questions on `fixed_hybrid`, including the
hierarchy question whose old judge calls exhausted the completion-token ceiling.

Quota history is retained in the ignored `.cache/` ledger. Historical budget
allowances are $0.1293392 for Gemini, $0.00288825 for the original Groq smoke,
and $0.02578395 for this stopped full attempt. Of the original $2.99 allowance,
$2.83198860 remains before the replacement smoke; its cost will also be deducted
before the replacement full run. Paid-rate estimates are not billed free-tier charges.
