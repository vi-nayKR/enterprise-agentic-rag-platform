# Replacement-run continuation

**Historical:** the user subsequently cancelled this same-model continuation.
Its processes were stopped before any full-run restart. The fixed same-model smoke
was moved to `results/attempts/groq_fixed_same_model_smoke/`; the new active smoke
is `results/smoke_groq_sambanova.json`, with independent SambaNova judging.

The source fixes are committed as `87738fb`; all 57 tests pass. The replacement
smoke uses the first five dev cases on `fixed_hybrid`, including the previously
truncating hierarchy question. Groq remains both answerer and judge. SambaNova
is deferred until the replacement full run finishes.

The replacement smoke is live in `results/smoke_groq.json` / `.md`. Daily-token
429s have delayed it. The running evaluator honors provider retry delays and
saves its quota/cost ledger. This is not a completed five-case result yet.

A separate one-shot continuation in the ignored `.cache/restart_fixed_groq.py`
waits for that smoke process to exit. It refuses to launch the full run unless
all five expected cases have valid answers and judgements, no errors, zero
truncations, zero HTTP 400s, and the expected provider/model/output limits.
It also refuses to overwrite an existing full-run output. A runnable
`--self-check` confirms incomplete/failed statuses cannot pass the gate.

After a successful gate, it writes `results/groq_smoke_summary.json` / `.md`
with provider p50/p95, queue and wall time, diagnostics, model/provider/date,
and paid-rate costs. Only then does it deduct the actual/reserved smoke cost
from the remaining $2.83198860 allowance and launch a fresh
`python -m src.evals.benchmark --split all` without `--resume`.

Continuation stdout/stderr: `.cache/fixed_groq_full.log` and
`.cache/fixed_groq_full.err.log`. These are local execution artifacts, not secrets.
If the smoke fails or the computer closes, do not assume the full run started;
inspect these files and `results/ablations_groq_all.json`. Resume an interrupted
smoke with `python -m src.evals.benchmark --smoke --resume`; after it passes,
use the same guarded continuation rather than bypassing its budget accounting.
Do not start a second evaluator concurrently or delete the quota ledger.
