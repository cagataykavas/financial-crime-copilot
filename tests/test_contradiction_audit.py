from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from contradiction_audit import ClaimConflictPolicy, audit_claim_conflicts
from copilot import (
    Copilot,
    Evidence,
    EvidenceClaim,
    EvidenceKind,
    FinancialCrimeCase,
    Severity,
    Signal,
)
from governance import PolicyGate, PolicyOutcome, validate_case_integrity


def _evidence(
    evidence_id: str,
    source: str,
    value: str | int | bool,
    *,
    effective_at: str = "2026-11-01T00:00:00+00:00",
    confidence: float = 0.95,
    claim_id: str | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        kind=EvidenceKind.PROFILE,
        source=source,
        summary="Synthetic profile evidence.",
        confidence=confidence,
        event_time=effective_at,
        claims=(
            EvidenceClaim(
                claim_id=claim_id or f"claim-{evidence_id}",
                subject_id="synthetic-subject",
                predicate="profile.risk_tier",
                value=value,
                effective_at=effective_at,
            ),
        ),
    )


def _monitoring_case(evidence: list[Evidence]) -> FinancialCrimeCase:
    transaction = Evidence(
        evidence_id="tx-context",
        kind=EvidenceKind.TRANSACTION,
        source="synthetic-transaction-source",
        summary="Synthetic transaction context is present.",
        confidence=1.0,
        event_time="2026-11-01T00:00:00+00:00",
    )
    signal = Signal(
        signal_id="sig-monitor",
        name="mild_velocity_change",
        severity=Severity.MEDIUM,
        score=0.55,
        rationale="Synthetic activity is elevated but below escalation thresholds.",
        evidence_ids=(transaction.evidence_id, evidence[0].evidence_id),
    )
    return FinancialCrimeCase(
        case_id="claim-audit-case",
        subject_id="synthetic-subject",
        subject_type="retail",
        opened_at="2026-11-01T00:00:00+00:00",
        evidence=[transaction, *evidence],
        signals=[signal],
        customer_impact=Severity.LOW,
    )


def test_independent_high_confidence_conflict_requires_human_review() -> None:
    case = _monitoring_case(
        [_evidence("profile-a", "source-a", "tier-a"), _evidence("profile-b", "source-b", "tier-b")]
    )

    recommendation = Copilot().recommend(case)
    decision = PolicyGate().evaluate(case, recommendation)

    assert recommendation.allowed_to_auto_execute
    assert decision.outcome is PolicyOutcome.REQUIRE_HUMAN
    assert "conflicting_claims_require_review" in decision.reasons
    assert len(decision.claim_conflicts) == 1
    assert decision.claim_conflicts[0].source_count == 2
    assert decision.claim_conflicts[0].value_count == 2


def test_report_is_content_bound_without_exposing_subject_or_values() -> None:
    case = _monitoring_case(
        [_evidence("profile-a", "source-a", "tier-a"), _evidence("profile-b", "source-b", "tier-b")]
    )
    report = audit_claim_conflicts(case)
    serialized = str(asdict(report))

    assert len(report.evidence_sha256) == 64
    assert "synthetic-subject" not in serialized
    assert "tier-a" not in serialized
    assert "tier-b" not in serialized


@pytest.mark.parametrize(
    "evidence",
    [
        [_evidence("a", "source-a", "same"), _evidence("b", "source-b", "same")],
        [
            _evidence("a", "source-a", "first", effective_at="2026-11-01T00:00:00+00:00"),
            _evidence("b", "source-b", "second", effective_at="2026-11-02T00:00:00+00:00"),
        ],
        [_evidence("a", "same-source", "first"), _evidence("b", "same-source", "second")],
        [
            _evidence("a", "source-a", "first", confidence=0.79),
            _evidence("b", "source-b", "second"),
        ],
    ],
)
def test_non_material_differences_do_not_create_conflicts(evidence) -> None:
    report = audit_claim_conflicts(_monitoring_case(evidence))

    assert report.findings == ()


def test_malformed_claim_metadata_blocks_execution() -> None:
    malformed = _evidence(
        "profile-a",
        "source-a",
        "tier-a",
        effective_at="2026-11-01T00:00:00",
    )
    case = _monitoring_case([malformed])

    decision = PolicyGate().evaluate(case, Copilot().recommend(case))

    assert decision.outcome is PolicyOutcome.BLOCK
    assert "case_integrity_failed" in decision.reasons
    assert any(issue.code == "invalid_claim_effective_at" for issue in decision.integrity_issues)


def test_duplicate_claim_ids_fail_integrity_validation() -> None:
    case = _monitoring_case(
        [
            _evidence("profile-a", "source-a", "tier-a", claim_id="same-claim"),
            _evidence("profile-b", "source-b", "tier-b", claim_id="same-claim"),
        ]
    )

    issues = validate_case_integrity(case)

    assert sum(issue.code == "duplicate_claim_id" for issue in issues) == 2


def test_policy_budgets_are_validated() -> None:
    with pytest.raises(ValueError, match="at least two"):
        replace(ClaimConflictPolicy(), min_independent_sources=1)
