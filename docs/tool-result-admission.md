# Tool-result provenance admission

An agent should not treat a syntactically valid tool response as trusted context.
Responses can be delayed, replayed, associated with the wrong call, returned by an
unexpected adapter, or swapped after transport. `service/tool_result_admission.py`
adds a deterministic gate at the boundary immediately before a result enters the
agent's context.

## Contract

The orchestrator first creates a `ToolCallIntent` with:

- the run, call and tool identities;
- a canonical SHA-256 digest of the arguments;
- the policy version and allowed result producers;
- a timezone-aware issue time and expiry.

After execution, the adapter returns a `ToolResultReceipt` that repeats the call
binding and adds the producer, attempt, status, execution interval and canonical
result digest. `ToolResultAdmissionLedger.admit()` compares the receipt with the
intent and the actual result bytes represented as canonical JSON.

Admission fails closed for identity or argument mismatch, an unauthorized producer,
unsuccessful status, result substitution, invalid attempt, expired or excessive
intent TTL, impossible timestamps, execution timeout, stale/future result, replay,
or a second result for an already admitted call. Resource budgets reject cyclic,
deep, oversized, non-JSON and non-finite values before hashing.

The in-process ledger locks the replay check and state transition together, so two
threads cannot both admit the same call. A rejected result does not consume the call;
the adapter may still deliver a valid, policy-bounded attempt.

## Example

```python
from datetime import UTC, datetime, timedelta

from service.tool_result_admission import (
    AdmissionPolicy,
    ToolResultAdmissionLedger,
    issue_result_receipt,
    prepare_tool_call,
)

now = datetime.now(UTC)
policy = AdmissionPolicy(policy_id="fc-tools-v1")
intent = prepare_tool_call(
    run_id="run-17",
    call_id="lookup-3",
    tool_name="case_lookup",
    arguments={"case_id": "FC-42"},
    policy=policy,
    allowed_producers=("case-service",),
    issued_at=now,
    expires_at=now + timedelta(seconds=30),
)

result = {"status": "ok", "evidence_ids": ["ev-9"]}
receipt = issue_result_receipt(
    intent,
    result=result,
    producer_id="case-service",
    attempt=1,
    status="success",
    started_at=now,
    completed_at=now + timedelta(milliseconds=40),
    policy=policy,
)

decision = ToolResultAdmissionLedger(policy).admit(
    intent,
    receipt,
    result=result,
    observed_at=now + timedelta(milliseconds=50),
)
assert decision.accepted
```

`AdmissionDecision` contains stable reason codes and content digests, not raw tool
arguments or result values. It can therefore be attached to the case audit trail
without duplicating sensitive financial-crime evidence.

## Trust boundary and limitations

This module proves binding and local single-use admission for the artifacts it is
given. It does **not** prove that the producer is honest, that the tool result is
factually correct, or that the adapter actually executed the declared tool. The
receipt is not signed.

The replay ledger is process-local and is lost on restart. A production deployment
must persist `(run_id, call_id)` admission with a transactional unique constraint or
compare-and-swap shared by every replica. It should authenticate adapters (for
example with workload identity and mTLS), sign or MAC receipts, bind schema/version
identities, and place the admission decision in the existing tamper-evident audit
chain. Payload values still need tool-specific schema and authorization validation.

The gate should run before any result is added to an LLM prompt, memory store,
planner state or side-effect decision. Running it only as an after-the-fact audit
does not enforce the boundary.
