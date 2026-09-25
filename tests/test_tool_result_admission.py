from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from service.tool_result_admission import (
    AdmissionPolicy,
    ArtifactError,
    ToolResultAdmissionLedger,
    issue_result_receipt,
    prepare_tool_call,
)

NOW = datetime(2026, 9, 25, 21, 0, tzinfo=UTC)
ARGS = {"case_id": "FC-42", "include": ["signals", "evidence"]}
RESULT = {"status": "ok", "evidence_ids": ["ev-2", "ev-1"]}


@pytest.fixture
def policy() -> AdmissionPolicy:
    return AdmissionPolicy(policy_id="tool-policy-v1")


def artifacts(policy: AdmissionPolicy):
    intent = prepare_tool_call(
        run_id="run-7",
        call_id="call-12",
        tool_name="case_lookup",
        arguments=ARGS,
        policy=policy,
        allowed_producers=("case-service",),
        issued_at=NOW,
        expires_at=NOW + timedelta(seconds=90),
    )
    receipt = issue_result_receipt(
        intent,
        result=RESULT,
        producer_id="case-service",
        attempt=1,
        status="success",
        started_at=NOW + timedelta(seconds=1),
        completed_at=NOW + timedelta(seconds=2),
        policy=policy,
    )
    return intent, receipt


def test_exact_bound_result_is_admitted_without_exposing_payload(policy: AdmissionPolicy) -> None:
    intent, receipt = artifacts(policy)

    decision = ToolResultAdmissionLedger(policy).admit(
        intent, receipt, result=RESULT, observed_at=NOW + timedelta(seconds=3)
    )

    assert decision.accepted is True
    assert decision.reason_codes == ()
    rendered = str(decision.to_dict())
    assert "FC-42" not in rendered
    assert "evidence_ids" not in rendered
    assert len(decision.evidence_digest) == 64


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("run_id", "another-run", "RUN_MISMATCH"),
        ("call_id", "another-call", "CALL_MISMATCH"),
        ("tool_name", "network_lookup", "TOOL_MISMATCH"),
        ("arguments_digest", "0" * 64, "ARGUMENTS_MISMATCH"),
        ("producer_id", "untrusted-adapter", "UNAUTHORIZED_PRODUCER"),
        ("status", "error", "UNSUCCESSFUL_RESULT"),
        ("attempt", 4, "ATTEMPT_OUT_OF_POLICY"),
    ],
)
def test_receipt_binding_and_policy_fail_closed(
    policy: AdmissionPolicy, field: str, value: object, code: str
) -> None:
    intent, receipt = artifacts(policy)

    decision = ToolResultAdmissionLedger(policy).admit(
        intent,
        replace(receipt, **{field: value}),
        result=RESULT,
        observed_at=NOW + timedelta(seconds=3),
    )

    assert decision.accepted is False
    assert code in decision.reason_codes


def test_swapped_result_is_rejected_and_does_not_consume_call(policy: AdmissionPolicy) -> None:
    intent, receipt = artifacts(policy)
    ledger = ToolResultAdmissionLedger(policy)

    rejected = ledger.admit(
        intent,
        receipt,
        result={"status": "ok", "evidence_ids": ["different"]},
        observed_at=NOW + timedelta(seconds=3),
    )
    accepted = ledger.admit(intent, receipt, result=RESULT, observed_at=NOW + timedelta(seconds=3))

    assert rejected.reason_codes == ("RESULT_DIGEST_MISMATCH",)
    assert accepted.accepted is True


@pytest.mark.parametrize(
    ("receipt_changes", "observed_delta", "code"),
    [
        ({"started_at": "2026-09-25T20:59:40Z"}, 3, "RECEIPT_BEFORE_INTENT"),
        ({"completed_at": "2026-09-25T21:00:00Z"}, 3, "RECEIPT_TIME_REVERSED"),
        ({"completed_at": "2026-09-25T21:01:10Z"}, 71, "EXECUTION_TIMEOUT"),
        ({"completed_at": "2026-09-25T21:00:20Z"}, 3, "RESULT_FROM_FUTURE"),
        ({}, 40, "RESULT_STALE"),
    ],
)
def test_temporal_failures_are_explicit(
    policy: AdmissionPolicy,
    receipt_changes: dict[str, object],
    observed_delta: int,
    code: str,
) -> None:
    intent, receipt = artifacts(policy)

    decision = ToolResultAdmissionLedger(policy).admit(
        intent,
        replace(receipt, **receipt_changes),
        result=RESULT,
        observed_at=NOW + timedelta(seconds=observed_delta),
    )

    assert decision.accepted is False
    assert code in decision.reason_codes


def test_expired_and_wrong_policy_intents_are_rejected(policy: AdmissionPolicy) -> None:
    intent, receipt = artifacts(policy)
    ledger = ToolResultAdmissionLedger(policy)

    expired = ledger.admit(intent, receipt, result=RESULT, observed_at=NOW + timedelta(seconds=100))
    wrong_policy = ledger.admit(
        replace(intent, policy_id="old-policy"),
        receipt,
        result=RESULT,
        observed_at=NOW + timedelta(seconds=3),
    )

    assert "INTENT_EXPIRED" in expired.reason_codes
    assert "POLICY_MISMATCH" in wrong_policy.reason_codes


def test_replay_and_second_result_for_call_are_rejected(policy: AdmissionPolicy) -> None:
    intent, receipt = artifacts(policy)
    ledger = ToolResultAdmissionLedger(policy)
    first = ledger.admit(intent, receipt, result=RESULT, observed_at=NOW + timedelta(seconds=3))
    replay = ledger.admit(intent, receipt, result=RESULT, observed_at=NOW + timedelta(seconds=3))
    second_receipt = replace(receipt, attempt=2)
    second = ledger.admit(
        intent, second_receipt, result=RESULT, observed_at=NOW + timedelta(seconds=3)
    )

    assert first.accepted is True
    assert replay.reason_codes == ("RECEIPT_REPLAYED", "CALL_ALREADY_ADMITTED")
    assert second.reason_codes == ("CALL_ALREADY_ADMITTED",)


def test_concurrent_admission_has_exactly_one_winner(policy: AdmissionPolicy) -> None:
    intent, receipt = artifacts(policy)
    ledger = ToolResultAdmissionLedger(policy)

    def admit_once() -> bool:
        return ledger.admit(
            intent, receipt, result=RESULT, observed_at=NOW + timedelta(seconds=3)
        ).accepted

    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(lambda _: admit_once(), range(64)))

    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 63


def test_canonical_digests_ignore_object_key_order(policy: AdmissionPolicy) -> None:
    first = prepare_tool_call(
        run_id="r",
        call_id="c",
        tool_name="t",
        arguments={"a": 1, "b": 2},
        policy=policy,
        allowed_producers=("p",),
        issued_at=NOW,
        expires_at=NOW + timedelta(seconds=5),
    )
    second = prepare_tool_call(
        run_id="r",
        call_id="c",
        tool_name="t",
        arguments={"b": 2, "a": 1},
        policy=policy,
        allowed_producers=("p",),
        issued_at=NOW,
        expires_at=NOW + timedelta(seconds=5),
    )
    assert first.arguments_digest == second.arguments_digest


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), {"x": object()}, {"x": 1, 2: "non-string-key"}],
)
def test_non_json_and_non_finite_arguments_are_rejected(
    policy: AdmissionPolicy, bad_value: object
) -> None:
    with pytest.raises(ArtifactError):
        prepare_tool_call(
            run_id="r",
            call_id="c",
            tool_name="t",
            arguments={"bad": bad_value},
            policy=policy,
            allowed_producers=("p",),
            issued_at=NOW,
            expires_at=NOW + timedelta(seconds=5),
        )


def test_cycles_depth_and_bytes_are_bounded() -> None:
    policy = AdmissionPolicy(
        policy_id="bounded", max_document_bytes=32, max_json_depth=2, max_json_nodes=20
    )
    cyclic: list[object] = []
    cyclic.append(cyclic)

    for value in (cyclic, {"a": {"b": {"c": 1}}}, {"value": "x" * 100}):
        with pytest.raises(ArtifactError):
            prepare_tool_call(
                run_id="r",
                call_id="c",
                tool_name="t",
                arguments=value,
                policy=policy,
                allowed_producers=("p",),
                issued_at=NOW,
                expires_at=NOW + timedelta(seconds=5),
            )


def test_policy_rejects_non_finite_and_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        AdmissionPolicy(policy_id="p", max_result_age_seconds=float("nan"))
    with pytest.raises(ValueError):
        AdmissionPolicy(policy_id="p", max_json_nodes=0)


def test_control_character_ids_are_rejected(policy: AdmissionPolicy) -> None:
    with pytest.raises(ArtifactError):
        prepare_tool_call(
            run_id="bad\nrun",
            call_id="c",
            tool_name="t",
            arguments={},
            policy=policy,
            allowed_producers=("p",),
            issued_at=NOW,
            expires_at=NOW + timedelta(seconds=5),
        )


def test_timezone_naive_intent_and_observation_are_rejected(policy: AdmissionPolicy) -> None:
    with pytest.raises(ArtifactError):
        prepare_tool_call(
            run_id="r",
            call_id="c",
            tool_name="t",
            arguments={},
            policy=policy,
            allowed_producers=("p",),
            issued_at=NOW.replace(tzinfo=None),
            expires_at=NOW + timedelta(seconds=5),
        )

    intent, receipt = artifacts(policy)
    decision = ToolResultAdmissionLedger(policy).admit(
        intent, receipt, result=RESULT, observed_at=NOW.replace(tzinfo=None)
    )
    assert decision.accepted is False
    assert "INVALID_RECEIPT" in decision.reason_codes


def test_evidence_is_deterministic_across_fresh_ledgers(policy: AdmissionPolicy) -> None:
    intent, receipt = artifacts(policy)
    decisions = [
        ToolResultAdmissionLedger(policy).admit(
            intent, receipt, result=RESULT, observed_at=NOW + timedelta(seconds=3)
        )
        for _ in range(2)
    ]
    assert decisions[0] == decisions[1]
