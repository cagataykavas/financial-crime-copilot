from __future__ import annotations

from copilot import (
    Action,
    Copilot,
    Evidence,
    EvidenceKind,
    FinancialCrimeCase,
    Severity,
    Signal,
    synthetic_case,
)
from governance import PolicyGate, PolicyOutcome, validate_case_integrity


def _clean_monitoring_case() -> FinancialCrimeCase:
    evidence = [
        Evidence(
            evidence_id="tx-clean",
            kind=EvidenceKind.TRANSACTION,
            source="synthetic-stream",
            summary="Transaction activity is mildly elevated.",
            confidence=0.98,
            event_time="2026-01-01T00:00:00+00:00",
        ),
        Evidence(
            evidence_id="profile-clean",
            kind=EvidenceKind.PROFILE,
            source="synthetic-profile",
            summary="Profile context is available and consistent.",
            confidence=0.95,
            event_time="2026-01-01T00:00:00+00:00",
        ),
    ]
    signal = Signal(
        signal_id="sig-monitor",
        name="mild_velocity_change",
        severity=Severity.MEDIUM,
        score=0.55,
        rationale="Synthetic activity is elevated but below escalation thresholds.",
        evidence_ids=("tx-clean", "profile-clean"),
    )
    return FinancialCrimeCase(
        case_id="monitor-clean",
        subject_id="subject-clean",
        subject_type="retail",
        opened_at="2026-01-01T00:00:00+00:00",
        evidence=evidence,
        signals=[signal],
        customer_impact=Severity.LOW,
    )


def test_missing_signal_evidence_blocks_execution() -> None:
    case = _clean_monitoring_case()
    case.signals[0] = Signal(
        signal_id="sig-monitor",
        name="mild_velocity_change",
        severity=Severity.MEDIUM,
        score=0.55,
        rationale="Broken provenance reference for test coverage.",
        evidence_ids=("does-not-exist",),
    )

    recommendation = Copilot().recommend(case)
    decision = PolicyGate().evaluate(case, recommendation)

    assert decision.outcome is PolicyOutcome.BLOCK
    assert "case_integrity_failed" in decision.reasons
    assert any(issue.code == "missing_signal_evidence" for issue in decision.integrity_issues)


def test_integrity_validator_detects_duplicate_ids_and_invalid_scores() -> None:
    case = _clean_monitoring_case()
    case.evidence.append(case.evidence[0])
    case.signals.append(
        Signal(
            signal_id="sig-invalid",
            name="invalid_score",
            severity=Severity.LOW,
            score=1.2,
            rationale="Out-of-range score for validation coverage.",
            evidence_ids=("tx-clean",),
        )
    )

    issues = validate_case_integrity(case)
    codes = {issue.code for issue in issues}

    assert "duplicate_evidence_id" in codes
    assert "invalid_signal_score" in codes


def test_clean_monitoring_action_can_pass_auto_policy() -> None:
    case = _clean_monitoring_case()
    recommendation = Copilot().recommend(case)
    decision = PolicyGate().evaluate(case, recommendation)

    assert recommendation.recommended_action is Action.CONTINUE_MONITORING
    assert recommendation.allowed_to_auto_execute is True
    assert decision.outcome is PolicyOutcome.ALLOW_AUTO
    assert decision.integrity_issues == ()


def test_material_escalation_requires_human_policy() -> None:
    case = synthetic_case()
    recommendation = Copilot().recommend(case)
    decision = PolicyGate().evaluate(case, recommendation)

    assert recommendation.recommended_action is Action.ESCALATE
    assert decision.outcome is PolicyOutcome.REQUIRE_HUMAN
    assert "material_disposition_requires_human_authorization" in decision.reasons
