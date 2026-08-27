from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from statistics import mean
from typing import ClassVar


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceKind(str, Enum):
    TRANSACTION = "transaction"
    PROFILE = "profile"
    NETWORK = "network"
    GEO = "geo"
    DOCUMENT = "document"
    RULE = "rule"


class Action(str, Enum):
    CLOSE = "close"
    REQUEST_INFORMATION = "request_information"
    ESCALATE = "escalate"
    RESTRICT = "restrict"
    CONTINUE_MONITORING = "continue_monitoring"


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    kind: EvidenceKind
    source: str
    summary: str
    confidence: float
    event_time: str
    contradictory: bool = False
    attributes: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Signal:
    signal_id: str
    name: str
    severity: Severity
    score: float
    rationale: str
    evidence_ids: tuple[str, ...]
    deterministic: bool = True


@dataclass
class FinancialCrimeCase:
    case_id: str
    subject_id: str
    subject_type: str
    opened_at: str
    evidence: list[Evidence] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    assigned_team: str = "financial-crime-review"
    sla_minutes: int = 60
    customer_impact: Severity = Severity.MEDIUM
    status: str = "open"
    reviewer_decision: Action | None = None
    reviewer_reason: str | None = None
    audit_events: list[dict[str, object]] = field(default_factory=list)

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {item.evidence_id: item for item in self.evidence}

    @property
    def contradictory_evidence_count(self) -> int:
        return sum(item.contradictory for item in self.evidence)

    @property
    def max_signal_score(self) -> float:
        return max((signal.score for signal in self.signals), default=0.0)


@dataclass(frozen=True)
class CopilotRecommendation:
    case_id: str
    recommended_action: Action
    priority: int
    headline: str
    narrative: str
    key_signal_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    missing_information: tuple[str, ...]
    uncertainty_notes: tuple[str, ...]
    allowed_to_auto_execute: bool


class CasePrioritizer:
    WEIGHTS: ClassVar[dict[Severity, int]] = {
        Severity.LOW: 8,
        Severity.MEDIUM: 18,
        Severity.HIGH: 32,
        Severity.CRITICAL: 45,
    }

    def priority(self, case: FinancialCrimeCase) -> int:
        score = 15
        score += self.WEIGHTS[case.customer_impact]
        score += round(30 * case.max_signal_score)
        score += min(15, case.contradictory_evidence_count * 6)
        if any(signal.severity is Severity.CRITICAL for signal in case.signals):
            score += 18
        elif any(signal.severity is Severity.HIGH for signal in case.signals):
            score += 5
        if case.sla_minutes <= 15:
            score += 12
        elif case.sla_minutes <= 30:
            score += 6
        return min(score, 100)


class Copilot:
    """Structured decision-support layer for synthetic financial-crime cases.

    The implementation is deliberately explicit. A production LLM could be used
    to draft the narrative, but evidence selection, action policy and audit
    logging remain deterministic and testable.
    """

    def __init__(self) -> None:
        self.prioritizer = CasePrioritizer()

    @staticmethod
    def _severity_rank(severity: Severity) -> int:
        return {
            Severity.LOW: 0,
            Severity.MEDIUM: 1,
            Severity.HIGH: 2,
            Severity.CRITICAL: 3,
        }[severity]

    def recommend(self, case: FinancialCrimeCase) -> CopilotRecommendation:
        if not case.signals:
            return CopilotRecommendation(
                case_id=case.case_id,
                recommended_action=Action.CLOSE,
                priority=self.prioritizer.priority(case),
                headline="No material risk signal detected",
                narrative=(
                    "The case contains no active risk signals. A reviewer should "
                    "verify that evidence ingestion completed successfully before closure."
                ),
                key_signal_ids=(),
                evidence_ids=tuple(item.evidence_id for item in case.evidence),
                missing_information=(),
                uncertainty_notes=("absence_of_signal_is_not_proof_of_absence",),
                allowed_to_auto_execute=False,
            )

        ranked = sorted(
            case.signals,
            key=lambda signal: (
                self._severity_rank(signal.severity),
                signal.score,
            ),
            reverse=True,
        )
        top = ranked[:3]
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for signal in top
                for evidence_id in signal.evidence_ids
            )
        )

        missing: list[str] = []
        evidence_kinds = {item.kind for item in case.evidence}
        if EvidenceKind.PROFILE not in evidence_kinds:
            missing.append("customer_profile_context")
        if EvidenceKind.TRANSACTION not in evidence_kinds:
            missing.append("transaction_context")
        if (
            any(signal.name == "geographic_inconsistency" for signal in top)
            and EvidenceKind.GEO not in evidence_kinds
        ):
            missing.append("geographic_context")

        uncertainty: list[str] = []
        if case.contradictory_evidence_count:
            uncertainty.append(
                f"{case.contradictory_evidence_count}_contradictory_evidence_item(s)"
            )
        low_confidence = [item for item in case.evidence if item.confidence < 0.70]
        if low_confidence:
            uncertainty.append(f"{len(low_confidence)}_low_confidence_evidence_item(s)")

        strongest = top[0]
        if missing:
            action = Action.REQUEST_INFORMATION
            headline = "Additional evidence required before disposition"
        elif strongest.severity is Severity.CRITICAL:
            action = Action.ESCALATE
            headline = "Critical signal requires specialist escalation"
        elif strongest.severity is Severity.HIGH or strongest.score >= 0.80:
            action = Action.ESCALATE
            headline = "High-risk pattern requires analyst review"
        elif strongest.severity is Severity.MEDIUM:
            action = Action.CONTINUE_MONITORING
            headline = "Moderate signal warrants continued review"
        else:
            action = Action.CLOSE
            headline = "Low-severity signal with supporting context"

        signal_summary = "; ".join(
            f"{signal.name} ({signal.severity.value}, {signal.score:.2f})"
            for signal in top
        )
        narrative = (
            f"The case was prioritized using {len(case.signals)} signal(s). "
            f"The strongest findings are: {signal_summary}. "
            f"The recommendation is '{action.value}'."
        )
        if uncertainty:
            narrative += " Reviewer attention is required because " + ", ".join(uncertainty) + "."

        # Material financial-crime disposition remains human-authorized in this demo.
        allowed_to_auto_execute = action in {Action.CONTINUE_MONITORING}

        return CopilotRecommendation(
            case_id=case.case_id,
            recommended_action=action,
            priority=self.prioritizer.priority(case),
            headline=headline,
            narrative=narrative,
            key_signal_ids=tuple(signal.signal_id for signal in top),
            evidence_ids=evidence_ids,
            missing_information=tuple(missing),
            uncertainty_notes=tuple(uncertainty),
            allowed_to_auto_execute=allowed_to_auto_execute,
        )

    @staticmethod
    def reviewer_decide(
        case: FinancialCrimeCase,
        recommendation: CopilotRecommendation,
        *,
        action: Action,
        reason: str,
        reviewer_id: str,
    ) -> None:
        case.reviewer_decision = action
        case.reviewer_reason = reason
        case.status = "resolved"
        case.audit_events.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "case_id": case.case_id,
                "reviewer_id": reviewer_id,
                "agent_recommendation": recommendation.recommended_action.value,
                "reviewer_action": action.value,
                "overridden": action != recommendation.recommended_action,
                "reason": reason,
                "evidence_ids": list(recommendation.evidence_ids),
                "signal_ids": list(recommendation.key_signal_ids),
            }
        )


@dataclass(frozen=True)
class QualityMetrics:
    cases: int
    agreement_rate: float
    override_rate: float
    average_priority: float
    evidence_coverage: float


def evaluate_reviews(
    rows: Iterable[tuple[FinancialCrimeCase, CopilotRecommendation]],
) -> QualityMetrics:
    pairs = list(rows)
    if not pairs:
        return QualityMetrics(0, 0.0, 0.0, 0.0, 0.0)

    resolved = [
        (case, rec)
        for case, rec in pairs
        if case.reviewer_decision is not None
    ]
    agreement_rate = (
        sum(case.reviewer_decision == rec.recommended_action for case, rec in resolved)
        / len(resolved)
        if resolved
        else 0.0
    )
    evidence_coverage = mean(
        1.0 if recommendation.evidence_ids else 0.0
        for _, recommendation in pairs
    )
    return QualityMetrics(
        cases=len(pairs),
        agreement_rate=agreement_rate,
        override_rate=1.0 - agreement_rate if resolved else 0.0,
        average_priority=mean(rec.priority for _, rec in pairs),
        evidence_coverage=evidence_coverage,
    )


def synthetic_case() -> FinancialCrimeCase:
    now = datetime.now(UTC).isoformat()
    evidence = [
        Evidence(
            evidence_id="tx-001",
            kind=EvidenceKind.TRANSACTION,
            source="synthetic_transaction_stream",
            summary="Five outbound transfers occurred within 18 minutes.",
            confidence=1.0,
            event_time=now,
            attributes={"count": 5, "window_minutes": 18, "total_amount": 48250},
        ),
        Evidence(
            evidence_id="net-001",
            kind=EvidenceKind.NETWORK,
            source="synthetic_graph_features",
            summary="Three counterparties are newly observed in the last 24 hours.",
            confidence=0.91,
            event_time=now,
            attributes={"new_counterparties": 3},
        ),
        Evidence(
            evidence_id="profile-001",
            kind=EvidenceKind.PROFILE,
            source="synthetic_customer_profile",
            summary="Observed activity is materially above the demo baseline.",
            confidence=0.94,
            event_time=now,
        ),
    ]
    signals = [
        Signal(
            signal_id="sig-velocity",
            name="transaction_velocity",
            severity=Severity.HIGH,
            score=0.86,
            rationale="Rapid sequence of outbound transfers exceeds synthetic baseline.",
            evidence_ids=("tx-001",),
        ),
        Signal(
            signal_id="sig-new-peers",
            name="new_counterparty_burst",
            severity=Severity.MEDIUM,
            score=0.72,
            rationale="Multiple previously unseen counterparties appeared in a short window.",
            evidence_ids=("net-001", "tx-001"),
        ),
        Signal(
            signal_id="sig-profile",
            name="profile_transaction_mismatch",
            severity=Severity.MEDIUM,
            score=0.69,
            rationale="Synthetic transaction intensity differs from the historical profile.",
            evidence_ids=("profile-001", "tx-001"),
        ),
    ]
    return FinancialCrimeCase(
        case_id="FC-DEMO-20418",
        subject_id="customer-demo-42",
        subject_type="retail",
        opened_at=now,
        evidence=evidence,
        signals=signals,
        sla_minutes=25,
        customer_impact=Severity.HIGH,
    )


def main() -> None:
    case = synthetic_case()
    copilot = Copilot()
    recommendation = copilot.recommend(case)
    print(recommendation)

    copilot.reviewer_decide(
        case,
        recommendation,
        action=Action.ESCALATE,
        reason="Reviewer agrees that the synthetic velocity pattern needs specialist review.",
        reviewer_id="demo-reviewer",
    )
    print("\nAudit event")
    print(case.audit_events[-1])


if __name__ == "__main__":
    main()
