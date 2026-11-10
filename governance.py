from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from copilot import Action, CopilotRecommendation, FinancialCrimeCase

MAX_RECOMMENDATION_IDS = 32
MAX_IDENTIFIER_BYTES = 128
MAX_HEADLINE_BYTES = 256
MAX_NARRATIVE_BYTES = 4_096


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


def validate_case_integrity(case: FinancialCrimeCase) -> tuple[IntegrityIssue, ...]:
    """Validate the provenance graph before a recommendation can drive execution."""
    issues: list[IntegrityIssue] = []

    evidence_ids = [item.evidence_id for item in case.evidence]
    signal_ids = [item.signal_id for item in case.signals]
    known_evidence = set(evidence_ids)

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
                        f"signal {signal.signal_id} score must be within [0, 1], got {signal.score}"
                    ),
                )
            )
        for evidence_id in signal.evidence_ids:
            if evidence_id not in known_evidence:
                issues.append(
                    IntegrityIssue(
                        code="missing_signal_evidence",
                        detail=(
                            f"signal {signal.signal_id} references missing evidence {evidence_id}"
                        ),
                    )
                )

    return tuple(issues)


def _valid_identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and value.isprintable()
        and len(value.encode("utf-8")) <= MAX_IDENTIFIER_BYTES
    )


def _validate_identifier_list(
    values: object,
    *,
    field: str,
) -> tuple[list[str], list[IntegrityIssue]]:
    if not isinstance(values, tuple):
        return [], [IntegrityIssue(f"invalid_{field}_container", f"{field} must be a tuple")]
    if len(values) > MAX_RECOMMENDATION_IDS:
        return [], [
            IntegrityIssue(
                f"too_many_{field}",
                f"{field} exceeds the {MAX_RECOMMENDATION_IDS}-item policy limit",
            )
        ]

    issues: list[IntegrityIssue] = []
    normalized: list[str] = []
    for index, value in enumerate(values):
        if not _valid_identifier(value):
            issues.append(
                IntegrityIssue(
                    f"invalid_{field}_identifier",
                    f"{field} contains an invalid identifier at position {index}",
                )
            )
            continue
        normalized.append(value)
    if _duplicates(normalized):
        issues.append(
            IntegrityIssue(
                f"duplicate_{field}",
                f"{field} must not contain duplicate identifiers",
            )
        )
    return normalized, issues


def validate_recommendation_integrity(
    case: FinancialCrimeCase,
    recommendation: CopilotRecommendation,
) -> tuple[IntegrityIssue, ...]:
    """Bind an agent recommendation to the exact case provenance graph."""
    issues: list[IntegrityIssue] = []

    if recommendation.case_id != case.case_id:
        issues.append(
            IntegrityIssue(
                "recommendation_case_mismatch",
                "recommendation case identity does not match the evaluated case",
            )
        )
    if not isinstance(recommendation.recommended_action, Action):
        issues.append(
            IntegrityIssue(
                "invalid_recommended_action",
                "recommended action is outside the governed action enum",
            )
        )
    if (
        isinstance(recommendation.priority, bool)
        or not isinstance(recommendation.priority, int)
        or not 0 <= recommendation.priority <= 100
    ):
        issues.append(
            IntegrityIssue(
                "invalid_recommendation_priority",
                "recommendation priority must be an integer within [0, 100]",
            )
        )
    for field, value, maximum in (
        ("headline", recommendation.headline, MAX_HEADLINE_BYTES),
        ("narrative", recommendation.narrative, MAX_NARRATIVE_BYTES),
    ):
        if (
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
            or not value.isprintable()
            or len(value.encode("utf-8")) > maximum
        ):
            issues.append(
                IntegrityIssue(
                    f"invalid_recommendation_{field}",
                    f"recommendation {field} violates the bounded text contract",
                )
            )

    signal_ids, signal_issues = _validate_identifier_list(
        recommendation.key_signal_ids,
        field="recommendation_signal_ids",
    )
    evidence_ids, evidence_issues = _validate_identifier_list(
        recommendation.evidence_ids,
        field="recommendation_evidence_ids",
    )
    issues.extend(signal_issues)
    issues.extend(evidence_issues)
    for field, values in (
        ("recommendation_missing_information", recommendation.missing_information),
        ("recommendation_uncertainty_notes", recommendation.uncertainty_notes),
    ):
        _normalized, field_issues = _validate_identifier_list(values, field=field)
        issues.extend(field_issues)

    known_signals = {signal.signal_id: signal for signal in case.signals}
    known_evidence = {evidence.evidence_id for evidence in case.evidence}
    selected_signals = [known_signals[item] for item in signal_ids if item in known_signals]

    if any(item not in known_signals for item in signal_ids):
        issues.append(
            IntegrityIssue(
                "unknown_recommendation_signal",
                "recommendation references a signal outside the case",
            )
        )
    if any(item not in known_evidence for item in evidence_ids):
        issues.append(
            IntegrityIssue(
                "unknown_recommendation_evidence",
                "recommendation references evidence outside the case",
            )
        )
    if case.signals and not signal_ids:
        issues.append(
            IntegrityIssue(
                "missing_recommendation_signals",
                "a recommendation for a signaled case must identify its key signals",
            )
        )

    required_evidence = {
        evidence_id for signal in selected_signals for evidence_id in signal.evidence_ids
    }
    if not case.signals:
        required_evidence = known_evidence
    if set(evidence_ids) != required_evidence:
        issues.append(
            IntegrityIssue(
                "recommendation_evidence_closure_mismatch",
                "recommendation evidence must exactly cover the selected signals",
            )
        )

    if not isinstance(recommendation.allowed_to_auto_execute, bool):
        issues.append(
            IntegrityIssue(
                "invalid_automatic_execution_flag",
                "automatic execution authority must be an explicit boolean",
            )
        )
    elif (
        recommendation.allowed_to_auto_execute
        and recommendation.recommended_action is not Action.CONTINUE_MONITORING
    ):
        issues.append(
            IntegrityIssue(
                "invalid_automatic_execution_claim",
                "only continued monitoring may be marked for automatic execution",
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
        issues = (
            *validate_case_integrity(case),
            *validate_recommendation_integrity(case, recommendation),
        )
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
