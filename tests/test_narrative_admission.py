from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from copilot import Action, Copilot, FinancialCrimeCase, Severity, synthetic_case
from service.narrative_admission import (
    NarrativeAdmissionError,
    NarrativePolicy,
    admit_narrative,
)

CONTEXT_SHA256 = "a" * 64


def _payload() -> dict:
    return {
        "schema_version": 1,
        "case_id": "FC-DEMO-20418",
        "action": "escalate",
        "context_sha256": CONTEXT_SHA256,
        "claims": [
            {
                "claim_id": "claim-1",
                "text": "Rapid outbound transfers support escalation.",
                "evidence_ids": ["tx-001"],
                "signal_ids": ["sig-velocity"],
            },
            {
                "claim_id": "claim-2",
                "text": "New counterparties add network risk context.",
                "evidence_ids": ["net-001", "tx-001"],
                "signal_ids": ["sig-new-peers"],
            },
            {
                "claim_id": "claim-3",
                "text": "Profile divergence provides additional support.",
                "evidence_ids": ["profile-001", "tx-001"],
                "signal_ids": ["sig-profile"],
            },
        ],
    }


def _admit(payload: dict | str | bytes, **kwargs):
    case = synthetic_case()
    recommendation = Copilot().recommend(case)
    raw = payload if isinstance(payload, (str, bytes)) else json.dumps(payload)
    return admit_narrative(
        raw,
        case,
        recommendation,
        expected_context_sha256=CONTEXT_SHA256,
        **kwargs,
    )


def _assert_rejected(payload: dict | str, code: str, **kwargs) -> None:
    with pytest.raises(NarrativeAdmissionError) as caught:
        _admit(payload, **kwargs)
    assert caught.value.code == code


def test_valid_narrative_is_deterministic_and_privacy_reduced() -> None:
    payload = _payload()

    first = _admit(payload)
    second = _admit(json.dumps(payload, indent=2, sort_keys=False))

    assert first == second
    assert [claim.claim_id for claim in first.claims] == ["claim-1", "claim-2", "claim-3"]
    assert first.receipt.claim_count == 3
    assert first.receipt.evidence_references == 3
    assert first.receipt.signal_references == 3
    encoded = json.dumps(asdict(first.receipt))
    assert "Rapid outbound" not in encoded
    assert "tx-001" not in encoded
    assert "FC-DEMO-20418" not in encoded


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("case_id", "other-case", "case_mismatch"),
        ("action", "close", "action_mismatch"),
        ("context_sha256", "b" * 64, "context_mismatch"),
    ],
)
def test_narrative_is_bound_to_case_action_and_context(field: str, value: str, code: str) -> None:
    payload = _payload()
    payload[field] = value

    _assert_rejected(payload, code)


def test_unknown_and_duplicate_citations_fail_closed() -> None:
    unknown = _payload()
    unknown["claims"][0]["evidence_ids"] = ["outside-context"]
    _assert_rejected(unknown, "unknown_evidence_reference")

    duplicate = _payload()
    duplicate["claims"][0]["evidence_ids"] = ["tx-001", "tx-001"]
    _assert_rejected(duplicate, "duplicate_reference")


def test_every_claim_is_cited_and_selected_context_is_fully_covered() -> None:
    uncited = _payload()
    uncited["claims"][0]["evidence_ids"] = []
    _assert_rejected(uncited, "uncited_claim")

    incomplete = _payload()
    incomplete["claims"] = incomplete["claims"][:2]
    _assert_rejected(incomplete, "incomplete_evidence_coverage")


def test_duplicate_json_fields_and_nonfinite_extensions_are_rejected() -> None:
    duplicate = json.dumps(_payload()).replace(
        '"schema_version": 1,',
        '"schema_version": 1, "schema_version": 1,',
        1,
    )
    _assert_rejected(duplicate, "duplicate_field")

    nonfinite = json.dumps(_payload()).replace('"schema_version": 1', '"schema_version": NaN')
    _assert_rejected(nonfinite, "invalid_json")


def test_control_characters_and_payload_budgets_are_rejected() -> None:
    unsafe = _payload()
    unsafe["claims"][0]["text"] = "Approved\u202efor display"
    _assert_rejected(unsafe, "unsafe_claim_text")

    _assert_rejected(
        _payload(), "payload_budget_exceeded", policy=NarrativePolicy(max_payload_bytes=50)
    )

    with pytest.raises(NarrativeAdmissionError) as caught:
        _admit(b"\xff".decode("latin-1").encode("latin-1"))
    assert caught.value.code == "invalid_encoding"


def test_recommendation_must_reference_existing_case_evidence() -> None:
    case = synthetic_case()
    recommendation = replace(Copilot().recommend(case), evidence_ids=("missing",))

    with pytest.raises(NarrativeAdmissionError) as caught:
        admit_narrative(
            json.dumps(_payload()),
            case,
            recommendation,
            expected_context_sha256=CONTEXT_SHA256,
        )

    assert caught.value.code == "recommendation_evidence_missing"


def test_ambiguous_case_identifiers_fail_closed() -> None:
    case = synthetic_case()
    case.evidence.append(case.evidence[0])
    recommendation = Copilot().recommend(case)

    with pytest.raises(NarrativeAdmissionError) as caught:
        admit_narrative(
            json.dumps(_payload()),
            case,
            recommendation,
            expected_context_sha256=CONTEXT_SHA256,
        )

    assert caught.value.code == "duplicate_case_evidence"


def test_no_evidence_recommendation_requires_a_deterministic_fallback() -> None:
    case = FinancialCrimeCase(
        case_id="empty",
        subject_id="subject",
        subject_type="retail",
        opened_at="2026-01-01T00:00:00+00:00",
        customer_impact=Severity.LOW,
    )
    recommendation = Copilot().recommend(case)
    payload = {
        "schema_version": 1,
        "case_id": "empty",
        "action": Action.CLOSE.value,
        "context_sha256": CONTEXT_SHA256,
        "claims": [
            {
                "claim_id": "claim-1",
                "text": "No evidence was supplied.",
                "evidence_ids": [],
                "signal_ids": [],
            }
        ],
    }

    with pytest.raises(NarrativeAdmissionError) as caught:
        admit_narrative(
            json.dumps(payload),
            case,
            recommendation,
            expected_context_sha256=CONTEXT_SHA256,
        )

    assert caught.value.code == "ungrounded_recommendation"
