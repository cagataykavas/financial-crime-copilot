from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

from copilot import Action, CopilotRecommendation, EvidenceKind, FinancialCrimeCase


@dataclass(frozen=True)
class IntegrityIssue:
    code: str
    detail: str
    blocking: bool = True


class PolicyOutcome(str, Enum):
    ALLOW_AUTO = "allow_auto"
    REQUIRE_HUMAN = "require_human"
    BLOCK = "block"


@dataclass(frozen=True)
class PolicyDecision:
    case_id: str
    recommended_action: Action
    outcome: PolicyOutcome
    reasons: tuple[str, ...]
    integrity_issues: tuple[IntegrityIssue, ...]


MAX_EVIDENCE_AGE = {
    EvidenceKind.TRANSACTION: timedelta(days=30),
    EvidenceKind.PROFILE: timedelta(days=365),
    EvidenceKind.NETWORK: timedelta(days=30),
    EvidenceKind.GEO: timedelta(days=90),
    EvidenceKind.DOCUMENT: timedelta(days=365),
    EvidenceKind.RULE: timedelta(days=30),
}
FUTURE_CLOCK_SKEW = timedelta(minutes=5)


MATERIAL_ACTIONS = {
    Action.CLOSE,
    Action.REQUEST_INFORMATION,
    Action.ESCALATE,
    Action.RESTRICT,
}


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def validate_case_integrity(case: FinancialCrimeCase) -> tuple[IntegrityIssue, ...]:
    """Validate the provenance graph before a recommendation can drive execution."""
    issues: list[IntegrityIssue] = []

    evidence_ids = [item.evidence_id for item in case.evidence]
    signal_ids = [item.signal_id for item in case.signals]
    known_evidence = set(evidence_ids)

    case_opened_at = _parse_timestamp(case.opened_at)
    if case_opened_at is None:
        issues.append(
            IntegrityIssue(
                code="invalid_case_opened_at",
                detail="case opened_at must be an ISO-8601 timestamp with a timezone",
            )
        )

    for evidence_id in sorted(_duplicates(evidence_ids)):
        issues.append(
            IntegrityIssue(
                code="duplicate_evidence_id",
                detail=f"evidence id appears more than once: {evidence_id}",
            )
        )

    for signal_id in sorted(_duplicates(signal_ids)):
        issues.append(
            IntegrityIssue(
                code="duplicate_signal_id",
                detail=f"signal id appears more than once: {signal_id}",
            )
        )

    for evidence in case.evidence:
        event_time = _parse_timestamp(evidence.event_time)
        if event_time is None:
            issues.append(
                IntegrityIssue(
                    code="invalid_evidence_event_time",
                    detail=(
                        f"evidence {evidence.evidence_id} event_time must be "
                        "an ISO-8601 timestamp with a timezone"
                    ),
                )
            )
        elif case_opened_at is not None:
            if event_time > case_opened_at + FUTURE_CLOCK_SKEW:
                issues.append(
                    IntegrityIssue(
                        code="future_dated_evidence",
                        detail=(
                            f"evidence {evidence.evidence_id} occurs after case opening "
                            "beyond the allowed clock skew"
                        ),
                    )
                )
            elif case_opened_at - event_time > MAX_EVIDENCE_AGE[evidence.kind]:
                issues.append(
                    IntegrityIssue(
                        code="stale_evidence",
                        detail=(
                            f"evidence {evidence.evidence_id} exceeds the "
                            f"{MAX_EVIDENCE_AGE[evidence.kind].days}-day freshness window"
                        ),
                        blocking=False,
                    )
                )

        if not 0.0 <= evidence.confidence <= 1.0:
            issues.append(
                IntegrityIssue(
                    code="invalid_evidence_confidence",
                    detail=(
                        f"evidence {evidence.evidence_id} confidence must be within [0, 1], "
                        f"got {evidence.confidence}"
                    ),
                )
            )

    for signal in case.signals:
        if not 0.0 <= signal.score <= 1.0:
            issues.append(
                IntegrityIssue(
                    code="invalid_signal_score",
                    detail=(
                        f"signal {signal.signal_id} score must be within [0, 1], "
                        f"got {signal.score}"
                    ),
                )
            )
        for evidence_id in signal.evidence_ids:
            if evidence_id not in known_evidence:
                issues.append(
                    IntegrityIssue(
                        code="missing_signal_evidence",
                        detail=(
                            f"signal {signal.signal_id} references missing evidence "
                            f"{evidence_id}"
                        ),
                    )
                )

    return tuple(issues)


class PolicyGate:
    """Keep recommendation logic separate from execution authority.

    Material dispositions always require a reviewer. The only action eligible for
    automatic execution in this reference implementation is continued monitoring,
    and even that is denied when provenance, uncertainty, or missing context is poor.
    """

    def evaluate(
        self,
        case: FinancialCrimeCase,
        recommendation: CopilotRecommendation,
    ) -> PolicyDecision:
        issues = validate_case_integrity(case)
        blocking = tuple(issue for issue in issues if issue.blocking)
        if blocking:
            return PolicyDecision(
                case_id=case.case_id,
                recommended_action=recommendation.recommended_action,
                outcome=PolicyOutcome.BLOCK,
                reasons=("case_integrity_failed",),
                integrity_issues=issues,
            )

        reasons: list[str] = []
        if recommendation.recommended_action in MATERIAL_ACTIONS:
            reasons.append("material_disposition_requires_human_authorization")
        if not recommendation.allowed_to_auto_execute:
            reasons.append("recommendation_not_marked_for_auto_execution")
        if recommendation.missing_information:
            reasons.append("missing_information_requires_review")
        if recommendation.uncertainty_notes:
            reasons.append("uncertainty_requires_review")
        if case.contradictory_evidence_count:
            reasons.append("contradictory_evidence_requires_review")
        if any(issue.code == "stale_evidence" for issue in issues):
            reasons.append("evidence_freshness_requires_review")

        if reasons:
            return PolicyDecision(
                case_id=case.case_id,
                recommended_action=recommendation.recommended_action,
                outcome=PolicyOutcome.REQUIRE_HUMAN,
                reasons=tuple(dict.fromkeys(reasons)),
                integrity_issues=issues,
            )

        return PolicyDecision(
            case_id=case.case_id,
            recommended_action=recommendation.recommended_action,
            outcome=PolicyOutcome.ALLOW_AUTO,
            reasons=("low_impact_monitoring_action_with_complete_evidence",),
            integrity_issues=issues,
        )
