# Queued post-run task: independent judge

User instruction received 2026-10-06. Do not implement or activate this until
the current Groq full run finishes. Keep its judge, responses, metrics, and
cost ledger unchanged. Do not read or commit API-key values in this document.

The user configured SambaNova credentials in the ignored `.env`:
`EVAL_JUDGE_BASE_URL`, `EVAL_JUDGE_MODEL=DeepSeek-V3.1`,
`EVAL_JUDGE_API_KEY`, and `EVAL_JUDGE_INPUT_USD_PER_MILLION` /
`EVAL_JUDGE_OUTPUT_USD_PER_MILLION`.

After the current run finishes:

1. Add optional `EVAL_JUDGE_*` settings, falling back to matching `EVAL_LLM_*`
   settings when unset. Use a separate judge client with its own conservative
   throttling and retries. Retain per-client ledgers and enforce the total
   evaluation spending cap across clients, passes, failures, and restarts.
2. Re-judge cached answers without generating or rewriting answers again.
   Preserve each question, reference answers, contexts, claims, citation IDs,
   and evidence fingerprint. Save independent predictions and costs in a
   separate results file; never overwrite the original Groq report.
3. Report original same-model versus cross-family judge scores, agreement,
   Cohen's kappa where defined, and disagreeing cases. Compare matching label
   identities on identical evidence and report missing/failed scores and
   denominators. Neither model's predictions replace human ground truth.
4. Evaluate both judges against the same frozen, independently human-labelled
   20-case worksheet. Human labels must remain blank/pending until supplied by
   the user; never infer labels from either judge. If still pending, report
   model-to-model agreement separately from human validation.
5. Smoke the independent re-judge on five cached cases first. SambaNova
   free-tier limits are unknown: throttle conservatively, honor provider
   retry guidance, back off on 429, and preserve checkpoints and costs.
6. Record provider, exact model ID, UTC evaluation date/time, prices, source
   report/evidence hashes, and generation-versus-judge provenance in results.
   Update README limitations from the measured comparison.

This task does not authorize production features before the existing ablation
checkpoint is reviewed. Do not rename the repository or change resume claims.

Current-run checkpoint at receipt: `results/ablations_groq_all.json` had
28 completed cases and stopped on a Groq HTTP 400 truncated structured judge
output for `fixed_hybrid:56e1c720e3433e140042316c`. Its generated answer was saved;
resume must reuse that answer and keep `openai/gpt-oss-120b` as the judge.

An unchanged retry reproduced the truncation. The case is retained as a failed
query with its error, answer, retrieval metrics and costs intact, and no judge
score. The full run continues under the original evaluation code and judge
settings; matching structured-output truncations are explicitly retained as
failures rather than scored or silently repaired. Results record this continuation
policy. Do not equate finalized failed cases with valid judge predictions.
