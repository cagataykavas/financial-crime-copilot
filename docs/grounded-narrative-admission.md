# Grounded LLM narrative admission

`service.narrative_admission.admit_narrative` is a fail-closed boundary between an untrusted
model response and text shown to a financial-crime reviewer. It accepts strict JSON rather than
free-form prose and binds the response to the deterministic recommendation that requested it.

The contract requires:

- the exact case, recommended action, and SHA-256 commitment of the model context;
- a bounded list of uniquely identified claims;
- at least one selected evidence citation for every claim;
- signal and evidence references drawn only from the recommendation;
- complete coverage of the recommendation's selected evidence and signals;
- bounded UTF-8 payload and claim sizes without control or invisible format characters.

Successful admission returns immutable validated claims for display and a separate privacy-reduced
receipt. The receipt contains only counts and commitments; it does not copy claim text, case IDs,
evidence IDs, or signal IDs, making that portion suitable for lower-sensitivity operational
telemetry.

```python
admission = admit_narrative(
    raw_model_json,
    case,
    recommendation,
    expected_context_sha256=context_digest,
)
display(admission.claims)
emit_telemetry(admission.receipt)
```

Admission does not grant execution authority; the existing deterministic `PolicyGate` remains the
only automation boundary. A rejected draft should be discarded and replaced with a deterministic
reviewer-facing fallback, not silently repaired or partially displayed.

## Limitations

Citation coverage proves structural provenance, not semantic entailment. A claim can cite relevant
evidence and still misrepresent it. The gate also does not establish source authority, model
identity, factual correctness, or resistance to a compromised prompt builder.

The next increment should add an independently evaluated claim-entailment stage with a reviewed
abstention threshold, while retaining this deterministic contract as the outer admission boundary.
