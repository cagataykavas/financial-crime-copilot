# Minimum-necessary LLM context

Financial-crime cases can contain customer identifiers, account details and free-text evidence. A
future narrative model should not receive the complete case object merely because the application
already has access to it. `service/prompt_context.py` adds a deterministic data-minimization boundary
between the governed case model and an external or internal LLM.

## Contract

`build_prompt_context()` accepts a case and its structured recommendation, then:

1. selects only evidence and signals explicitly referenced by the recommendation;
2. replaces case, subject, evidence and signal identifiers before model egress;
3. includes only policy-allowlisted scalar evidence attributes;
4. excludes evidence summaries, signal rationales and source-system labels by default;
5. requires timezone-aware evidence timestamps and normalizes them to UTC;
6. bounds evidence, signals, attributes, strings and the exact canonical context size;
7. records an explicit projection-policy ID for audit and rollout control;
8. emits a SHA-256 digest and a local-only alias manifest for later citation resolution.

The model-facing object carries `trust="untrusted_case_evidence"`. That label is an integration
contract: evidence remains data, not an instruction source.

## Pseudonymization and aliases

Case and subject references use HMAC-SHA-256 with three inputs:

```text
secret key + purpose + identifier
```

The purpose is domain separation. The same subject receives a stable reference within one approved
use, while different purposes cannot be joined by comparing references. Evidence and signal IDs are
replaced with compact `e1` / `s1` aliases. Their raw mappings remain only in `ContextManifest`.

A plain hash is deliberately not used: low-entropy or enumerable customer identifiers are vulnerable
to dictionary attacks. The caller must supply at least 32 bytes of key material and a non-secret
`key_id`; the secret is never stored in the returned bundle.

## Example

```python
import os

from copilot import Copilot, synthetic_case
from service.prompt_context import build_prompt_context

case = synthetic_case()
recommendation = Copilot().recommend(case)
bundle = build_prompt_context(
    case,
    recommendation,
    pseudonym_key=os.environ["PROMPT_PSEUDONYM_KEY"].encode(),
    key_id="prompt-hmac-2026-09",
)

# Send only this JSON to the model provider.
model_json = bundle.canonical_model_json

# Keep this mapping inside the governed service for citation resolution/audit.
local_manifest = bundle.manifest
```

Default allowlists retain structured decision features such as transaction count, time window,
currency, total amount and network-risk aggregates. Fields such as names, emails, account numbers,
addresses and arbitrary nested objects are omitted. A deployment can replace every per-kind allowlist
with a narrower one and can disable event timestamps.

## Fail-closed behavior

Projection is rejected when:

- the recommendation belongs to another case;
- selected evidence or signals are duplicated or missing;
- a selected signal would cite evidence omitted from model context;
- an allowlisted attribute is nested, non-finite or oversized;
- evidence confidence or signal scores are invalid;
- count or serialized-byte budgets are exceeded;
- HMAC key or key identifier requirements are not met.

Rejections expose stable codes without including raw customer values.

## Security and methodology limits

- Data minimization is not anonymization. Event time, amounts and rare attribute combinations may
  still permit re-identification. Policies should be reviewed against the actual threat model and
  jurisdiction.
- HMAC pseudonyms remain linkable inside one purpose. Rotate and version keys, restrict manifest
  access and define retention periods.
- The local manifest contains raw IDs and must never be serialized into the provider request, model
  logs or prompt telemetry.
- Allowlists reduce disclosure; they do not establish that a field is accurate or authorized for a
  particular analyst. Case access control must run before projection.
- The trust label does not detect prompt injection in string values. Prompt construction must keep
  evidence structurally separated from system/developer instructions.
- SHA-256 identifies the exact projected bytes; it does not authenticate upstream evidence.

The next step is an adapter that records the context digest, model/version, prompt-template version
and cited aliases in the existing tamper-evident reviewer audit chain.
