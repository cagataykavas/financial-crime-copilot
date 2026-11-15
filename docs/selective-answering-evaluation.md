# Selective-answering release gate

An evidence-grounded copilot should abstain when its evidence is insufficient,
but a system that abstains on every case is not useful. This module evaluates a
**fixed** confidence threshold against reviewed, held-out observations and gates
both sides of that trade-off.

## Evidence contract

Each observation contains exactly:

```json
{
  "case_id": "held-out-0001",
  "confidence": 0.91,
  "answerable": true,
  "answer_supported": true
}
```

- `answerable` is a reviewed label stating whether the admitted evidence is
  sufficient to answer the case prompt.
- `answer_supported` states whether the candidate answer is supported by that
  evidence. An unanswerable case cannot be marked supported.
- `confidence` is the candidate system's pre-threshold score. It must be finite
  and bounded to `[0, 1]`.
- `case_id` must be unique so repeated cases cannot inflate the evaluation.

The confidence threshold and all release limits must be chosen before evaluating
the held-out set. Searching thresholds on the same records invalidates the gate.

## Metrics and decision

The gate reports three binomial metrics:

| Metric | Numerator / denominator | Release bound |
|---|---|---|
| Selective risk | unsupported answered / all answered | Wilson upper bound ≤ policy maximum |
| False-answer rate | answered unanswerable / all unanswerable | Wilson upper bound ≤ policy maximum |
| Answerable coverage | answered answerable / all answerable | Wilson lower bound ≥ policy minimum |

Using interval bounds prevents a small sample with zero observed failures from
passing on an optimistic point estimate. Minimum total, answerable and
unanswerable sample sizes are enforced separately.

The report includes a SHA-256 digest over the canonicalized observations, policy
and schema version. The digest binds evidence; it does not authenticate who
produced or reviewed it.

## CLI

The input can be a JSON array or an object containing `observations`.

```bash
python -m service.selective_answering reviewed_outputs.json \
  --output artifacts/selective_answering.json \
  --confidence-threshold 0.75
```

The report is written atomically. Exit codes are:

- `0`: policy passed;
- `2`: admitted evidence failed one or more release limits;
- `3`: malformed or inadmissible evidence.

## Limitations

This gate assumes the reviewed labels, case sampling and confidence values are
trustworthy and point-in-time correct. It does not detect benchmark leakage,
distribution shift, reviewer disagreement, segment-level regressions, prompt
injection, citation fabrication or policy harm. Confidence need not be calibrated
merely because thresholded selective metrics pass. A production release should
also evaluate slices, calibration, grounding, security and delayed outcomes.
