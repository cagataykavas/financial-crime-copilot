# Context-order robustness gate

RAG systems can produce different decisions when the same retrieved chunks are
reordered. This is especially risky when retrieval scores are close or a prompt
builder changes how it packs context. `service.context_order_evaluation` audits
recorded model trials without calling a model or claiming synthetic quality
results.

Each JSONL row captures one completed evaluation trial:

```json
{"case_id":"case-17","trial_id":"baseline","is_baseline":true,"evidence_order":["ev-1","ev-2"],"action":"escalate","cited_evidence":["ev-1","ev-2"]}
{"case_id":"case-17","trial_id":"reverse","is_baseline":false,"evidence_order":["ev-2","ev-1"],"action":"escalate","cited_evidence":["ev-2","ev-1"]}
```

Run the gate with:

```bash
python -m service.context_order_evaluation trials.jsonl \
  --minimum-trials 2 \
  --minimum-citation-jaccard 1.0 \
  --output artifacts/context-order-report.json
```

Exit `0` means every case passed; malformed evidence or a failed case exits `2`.
The report is deterministic and written atomically.

## What is enforced

- exactly one baseline and the configured minimum trials per case;
- unique trial and evidence IDs;
- identical evidence membership between baseline and variants, isolating order
  as the tested variable;
- at least one real reordering rather than a duplicate baseline;
- an unchanged structured action;
- citation-set Jaccard at or above the configured threshold;
- citations that refer only to evidence present in that trial;
- bounded ingestion: 4 MB, 5,000 trials, and 256 evidence items per trial,
  with duplicate keys, non-finite numbers, blank lines, and symlink inputs
  rejected.

## Interpretation and limitations

This evaluates outputs already produced by a generation pipeline. It does not
invoke an LLM, judge narrative semantics, prove citation entailment, or replace
the separate evidence-grounding controls. Exact action stability is deliberate
for regulated case dispositions; citation Jaccard can be relaxed for exploratory
workloads.

The next increment should generate seeded permutations through the production
prompt builder and retain model, prompt-template, decoding, and corpus-snapshot
identifiers alongside these trials.
