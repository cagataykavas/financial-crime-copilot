"""Evaluate alert-ranking quality under a finite human-review budget."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import ceil, isfinite


@dataclass(frozen=True)
class AlertOutcome:
    alert_id: str
    risk_score: float
    confirmed_positive: bool


@dataclass(frozen=True)
class CapacityPolicy:
    review_fraction: float = 0.20
    min_alerts: int = 20
    min_positives: int = 3
    min_precision_at_capacity: float = 0.50
    min_recall_at_capacity: float = 0.50
    min_lift_over_random: float = 2.0


DEFAULT_POLICY = CapacityPolicy()


@dataclass(frozen=True)
class CapacityReport:
    passed: bool
    reasons: tuple[str, ...]
    total_alerts: int
    total_positives: int
    review_count: int
    reviewed_positives: int
    missed_positives: int
    precision_at_capacity: float
    recall_at_capacity: float
    base_rate: float
    lift_over_random: float
    workload_reduction: float
    reviewed_alert_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        result["reviewed_alert_ids"] = list(self.reviewed_alert_ids)
        return result


def evaluate_review_capacity(
    outcomes: Iterable[AlertOutcome], policy: CapacityPolicy = DEFAULT_POLICY
) -> CapacityReport:
    """Measure ranking utility at a fixed review fraction and apply a release gate."""
    rows = tuple(outcomes)
    _validate_policy(policy)
    _validate_outcomes(rows)

    total = len(rows)
    positives = sum(row.confirmed_positive for row in rows)
    review_count = min(total, max(1, ceil(total * policy.review_fraction)))
    ranked = sorted(rows, key=lambda row: (-row.risk_score, row.alert_id))
    reviewed = ranked[:review_count]
    reviewed_positives = sum(row.confirmed_positive for row in reviewed)
    precision = reviewed_positives / review_count
    recall = reviewed_positives / positives
    base_rate = positives / total
    lift = precision / base_rate
    reasons: list[str] = []
    if precision < policy.min_precision_at_capacity:
        reasons.append("precision_below_floor")
    if recall < policy.min_recall_at_capacity:
        reasons.append("recall_below_floor")
    if lift < policy.min_lift_over_random:
        reasons.append("lift_below_floor")

    return CapacityReport(
        passed=not reasons,
        reasons=tuple(reasons),
        total_alerts=total,
        total_positives=positives,
        review_count=review_count,
        reviewed_positives=reviewed_positives,
        missed_positives=positives - reviewed_positives,
        precision_at_capacity=precision,
        recall_at_capacity=recall,
        base_rate=base_rate,
        lift_over_random=lift,
        workload_reduction=1.0 - review_count / total,
        reviewed_alert_ids=tuple(row.alert_id for row in reviewed),
    )


def _validate_policy(policy: CapacityPolicy) -> None:
    fractions = {
        "review_fraction": policy.review_fraction,
        "min_precision_at_capacity": policy.min_precision_at_capacity,
        "min_recall_at_capacity": policy.min_recall_at_capacity,
    }
    for name, value in fractions.items():
        if not isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be finite and in (0, 1]")
    if policy.min_alerts < 1:
        raise ValueError("min_alerts must be positive")
    if policy.min_positives < 1:
        raise ValueError("min_positives must be positive")
    if not isfinite(policy.min_lift_over_random) or policy.min_lift_over_random <= 0:
        raise ValueError("min_lift_over_random must be finite and positive")


def _validate_outcomes(rows: tuple[AlertOutcome, ...]) -> None:
    if not rows:
        raise ValueError("outcomes must not be empty")
    ids: set[str] = set()
    for row in rows:
        if not row.alert_id or not row.alert_id.strip():
            raise ValueError("alert_id must not be blank")
        if row.alert_id in ids:
            raise ValueError(f"duplicate alert_id: {row.alert_id}")
        ids.add(row.alert_id)
        if not isfinite(row.risk_score):
            raise ValueError(f"risk_score must be finite: {row.alert_id}")


def validate_evidence_size(
    outcomes: Iterable[AlertOutcome], policy: CapacityPolicy = DEFAULT_POLICY
) -> tuple[AlertOutcome, ...]:
    """Fail closed when delayed outcomes do not support stable evaluation."""
    rows = tuple(outcomes)
    _validate_policy(policy)
    _validate_outcomes(rows)
    if len(rows) < policy.min_alerts:
        raise ValueError(f"at least {policy.min_alerts} alerts are required")
    positives = sum(row.confirmed_positive for row in rows)
    if positives < policy.min_positives:
        raise ValueError(f"at least {policy.min_positives} positives are required")
    return rows


def evaluate_sufficient_review_capacity(
    outcomes: Iterable[AlertOutcome], policy: CapacityPolicy = DEFAULT_POLICY
) -> CapacityReport:
    """Validate evidence sufficiency, then evaluate operational ranking quality."""
    return evaluate_review_capacity(validate_evidence_size(outcomes, policy), policy)
