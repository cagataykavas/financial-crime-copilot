# Counterfactual evidence-sensitivity audit

A model can attach valid evidence IDs to a claim without using that evidence to form the claim.
`service.evidence_sensitivity` adds an offline behavioral gate: remove one evidence item from the
model context, rerun generation with the same reviewed configuration, and compare stable claim IDs.

For each trial, the audit requires the baseline claims, the removed evidence ID, the evidence that
remained available, and the counterfactual claims. It then checks that:

- a claim supported only by removed evidence is withdrawn;
- a claim with alternative baseline support is withdrawn or rebound only to that support;
- claims unrelated to the ablation remain present with unchanged citations;
- the counterfactual output does not cite removed or unavailable evidence;
- the counterfactual output does not introduce new claims; and
- the suite contains enough affected, uniquely supported, and unaffected control claims.

The default policy requires 20 trials, 20 affected claims, 10 uniquely supported claims, and 20
unaffected controls. It requires at least 95% withdrawal for uniquely supported claims, 100%
appropriate adjustment across all affected claims, and 90% retention of unaffected claims.

Input is a strict JSON array. Claim IDs must represent the same pre-registered proposition across
the baseline and counterfactual run; an omitted claim is treated as withdrawn.

```json
[
  {
    "trial_id": "opaque-trial-001",
    "ablated_evidence_id": "transaction-17",
    "retained_evidence_ids": ["profile-4", "network-9"],
    "baseline_claims": [
      {"claim_id": "claim-velocity", "evidence_ids": ["transaction-17"]},
      {"claim_id": "claim-network", "evidence_ids": ["network-9"]}
    ],
    "counterfactual_claims": [
      {"claim_id": "claim-network", "evidence_ids": ["network-9"]}
    ]
  }
]
```

```bash
python -m service.evidence_sensitivity ablation-trials.json \
  --output evidence-sensitivity-report.json
```

Exit code `0` means the suite passed, `2` means valid evaluation evidence failed policy, and `3`
means the evidence or operation was invalid. Parsing rejects duplicate keys, unknown fields,
non-finite values, symlinks, oversized inputs, duplicate identifiers, undeclared baseline
citations, and excessive work. Reports are written through an fsynced temporary file and atomic
replacement. They expose only bounded hashed trial/claim references; a SHA-256 digest binds the
complete input and policy.

## Limitations

This is a structural and behavioral sensitivity test, not a semantic-entailment judge. Stable
claim IDs must be assigned by a reviewed adapter; changing claim segmentation can change the
result. Evidence ablation may produce distribution shift, and passing does not establish source
authority, factual correctness, retrieval completeness, or causal identification. Hashing opaque
identifiers reduces accidental disclosure but is not an anonymization guarantee.

The next step is a blinded evaluation that combines this deterministic gate with human-reviewed
entailment labels and multiple-evidence interventions, then measures uncertainty across cases,
models, and prompt versions.
