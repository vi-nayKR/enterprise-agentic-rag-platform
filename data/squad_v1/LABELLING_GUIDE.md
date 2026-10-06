# Judge calibration: your labels only

The evaluator will create `labels_todo.jsonl` from actual model answers to dev
questions. It must contain blank labels, not the model's predicted labels.
Do not consult judge outputs while labelling; that would bias the comparison.

For each row, read the question, reference answers, retrieved contexts, and model
claims. Set every null label to a JSON boolean (`true` or `false`):

- `answer_relevant`: does the answer directly and correctly answer the question?
  Reference answers help here. An unnecessary abstention on this answerable
  subset is false; unrelated copied context is false.
- Each `claim_support.supported`: is the entire claim supported by the supplied
  retrieved contexts? Use only those contexts, not your background knowledge or
  the reference answer. A fact can be correct but unsupported by retrieval.
- Each `citation_support.supported`: does that exact cited chunk support the
  corresponding claim? A missing/invented chunk ID is false. Evidence elsewhere
  in the context cannot rescue an incorrect citation.

Add your name in `reviewer` and explain ambiguous or partial claims in `notes`.
For a partially supported claim, use false and note the unsupported part.
Keep the source text, claims, IDs, and evidence hashes unchanged. Missing labels
must remain pending; the agent must not guess or fill them on your behalf.

Faithfulness is the fraction of supported claims; citation accuracy is the
fraction of supported citation/claim pairs. Report answer relevance and judge
agreement separately. For abstentions, claim faithfulness and citation accuracy
are undefined rather than automatically perfect. Agreement and Cohen's kappa
must not be published before all human labels are complete.
