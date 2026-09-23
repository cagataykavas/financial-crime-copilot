from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from copilot import CopilotRecommendation, Evidence, EvidenceKind, FinancialCrimeCase, Signal

DEFAULT_ALLOWED_ATTRIBUTES: dict[EvidenceKind, frozenset[str]] = {
    EvidenceKind.TRANSACTION: frozenset({"count", "currency", "total_amount", "window_minutes"}),
    EvidenceKind.PROFILE: frozenset({"risk_tier", "tenure_days"}),
    EvidenceKind.NETWORK: frozenset({"new_counterparties", "network_risk_band"}),
    EvidenceKind.GEO: frozenset({"country_risk_band", "distance_km"}),
    EvidenceKind.DOCUMENT: frozenset({"document_type", "verification_status"}),
    EvidenceKind.RULE: frozenset({"rule_version", "threshold"}),
}


class ContextProjectionError(ValueError):
    """A stable fail-closed rejection from the model-context boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True)
class PromptContextPolicy:
    """Minimum-necessary projection policy for one model purpose."""

    purpose: str = "draft_case_narrative"
    policy_id: str = "minimum-necessary-v1"
    allowed_attributes: dict[EvidenceKind, frozenset[str]] = field(
        default_factory=lambda: dict(DEFAULT_ALLOWED_ATTRIBUTES)
    )
    include_event_time: bool = True
    max_evidence: int = 20
    max_signals: int = 10
    max_attributes_per_evidence: int = 10
    max_string_bytes: int = 512
    max_context_bytes: int = 16_384

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]{2,63}", self.purpose
        ):
            raise ValueError("purpose must be a lowercase policy identifier")
        if not isinstance(self.policy_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,64}", self.policy_id
        ):
            raise ValueError("policy_id must be a safe non-empty identifier")
        limits = {
            "max_evidence": self.max_evidence,
            "max_signals": self.max_signals,
            "max_attributes_per_evidence": self.max_attributes_per_evidence,
            "max_string_bytes": self.max_string_bytes,
            "max_context_bytes": self.max_context_bytes,
        }
        for name, value in limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for kind, names in self.allowed_attributes.items():
            if not isinstance(kind, EvidenceKind):
                raise TypeError("allowed attribute policies must use EvidenceKind keys")
            if not isinstance(names, frozenset) or any(
                not isinstance(name, str) or not name for name in names
            ):
                raise ValueError("allowed attribute names must be non-empty frozen strings")


@dataclass(frozen=True)
class ContextManifest:
    """Local-only provenance that must not be sent to the model provider."""

    key_id: str
    policy_id: str
    case_id: str
    context_sha256: str
    context_bytes: int
    evidence_refs: tuple[tuple[str, str], ...]
    signal_refs: tuple[tuple[str, str], ...]
    excluded_attributes: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class PromptContextBundle:
    canonical_model_json: str
    manifest: ContextManifest

    @property
    def model_context(self) -> dict[str, Any]:
        value = json.loads(self.canonical_model_json)
        if not isinstance(value, dict):  # Defensive invariant.
            raise TypeError("canonical model context is not an object")
        return value


def _reject(code: str, detail: str) -> None:
    raise ContextProjectionError(code, detail)


def _unique_index(
    items: list[Any],
    *,
    id_field: str,
    duplicate_code: str,
    max_id_bytes: int,
) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for item in items:
        item_id = _require_bounded_string(
            getattr(item, id_field),
            label=id_field,
            max_bytes=max_id_bytes,
        )
        if item_id in index:
            _reject(duplicate_code, f"case contains duplicate {id_field}")
        index[item_id] = item
    return index


def _require_bounded_string(value: Any, *, label: str, max_bytes: int) -> str:
    if not isinstance(value, str):
        _reject("invalid_string", f"{label} must be a string")
    size = len(value.encode("utf-8"))
    if size > max_bytes:
        _reject("string_too_large", f"{label} is {size} bytes; limit is {max_bytes}")
    return value


def _scalar_attribute(value: object, *, label: str, max_string_bytes: int) -> object:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _reject("non_finite_attribute", f"{label} contains a non-finite number")
        return value
    if isinstance(value, str):
        return _require_bounded_string(value, label=label, max_bytes=max_string_bytes)
    _reject("non_scalar_attribute", f"{label} must be a JSON scalar")


def _probability(value: object, *, code: str, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        _reject(code, f"{label} must be finite and within [0, 1]")
    return float(value)


def _normalize_timestamp(value: object, *, label: str, max_bytes: int) -> str:
    timestamp = _require_bounded_string(value, label=label, max_bytes=max_bytes)
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        _reject("invalid_event_time", f"{label} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _reject("invalid_event_time", f"{label} must include a timezone")
    return parsed.astimezone(UTC).isoformat()


def _validated_ids(values: tuple[str, ...], *, label: str, max_bytes: int) -> tuple[str, ...]:
    return tuple(
        _require_bounded_string(value, label=label, max_bytes=max_bytes) for value in values
    )


def _pseudonym(*, secret: bytes, purpose: str, label: str, value: str) -> str:
    message = f"{purpose}\0{label}\0{value}".encode()
    return hmac.new(secret, message, hashlib.sha256).hexdigest()[:32]


def _project_evidence(
    evidence: Evidence,
    *,
    ref: str,
    policy: PromptContextPolicy,
) -> tuple[dict[str, Any], int]:
    allowed = policy.allowed_attributes.get(evidence.kind, frozenset())
    included_names = sorted(set(evidence.attributes) & allowed)
    if len(included_names) > policy.max_attributes_per_evidence:
        _reject(
            "attribute_budget_exceeded",
            f"evidence {ref} exceeds the allowed attribute count",
        )
    attributes = {
        name: _scalar_attribute(
            evidence.attributes[name],
            label=f"evidence {ref} attribute",
            max_string_bytes=policy.max_string_bytes,
        )
        for name in included_names
    }
    confidence = _probability(
        evidence.confidence,
        code="invalid_evidence_confidence",
        label=f"evidence {ref} confidence",
    )
    projected: dict[str, Any] = {
        "ref": ref,
        "kind": evidence.kind.value,
        "confidence": confidence,
        "contradictory": evidence.contradictory,
        "attributes": attributes,
    }
    if policy.include_event_time:
        projected["event_time"] = _normalize_timestamp(
            evidence.event_time,
            label=f"evidence {ref} event time",
            max_bytes=policy.max_string_bytes,
        )
    return projected, len(evidence.attributes) - len(attributes)


def _project_signal(
    signal: Signal,
    *,
    ref: str,
    evidence_aliases: dict[str, str],
    max_string_bytes: int,
) -> dict[str, Any]:
    score = _probability(
        signal.score,
        code="invalid_signal_score",
        label=f"signal {ref} score",
    )
    signal_evidence_ids = _validated_ids(
        signal.evidence_ids,
        label=f"signal {ref} evidence id",
        max_bytes=max_string_bytes,
    )
    unknown = [item for item in signal_evidence_ids if item not in evidence_aliases]
    if unknown:
        _reject(
            "signal_evidence_not_selected",
            f"signal {ref} references evidence outside the recommendation",
        )
    return {
        "ref": ref,
        "name": _require_bounded_string(
            signal.name,
            label=f"signal {ref} name",
            max_bytes=max_string_bytes,
        ),
        "severity": signal.severity.value,
        "score": score,
        "evidence_refs": [evidence_aliases[item] for item in signal_evidence_ids],
    }


def build_prompt_context(
    case: FinancialCrimeCase,
    recommendation: CopilotRecommendation,
    *,
    pseudonym_key: bytes,
    key_id: str,
    policy: PromptContextPolicy | None = None,
) -> PromptContextBundle:
    """Project a case into bounded, purpose-scoped, minimum-necessary model data."""

    active_policy = policy or PromptContextPolicy()
    if not isinstance(pseudonym_key, bytes) or len(pseudonym_key) < 32:
        raise ValueError("pseudonym_key must contain at least 32 bytes")
    if not isinstance(key_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", key_id):
        raise ValueError("key_id must be a safe non-empty identifier")
    if recommendation.case_id != case.case_id:
        _reject("case_mismatch", "recommendation does not belong to the supplied case")
    case_id = _require_bounded_string(
        case.case_id,
        label="case id",
        max_bytes=active_policy.max_string_bytes,
    )
    subject_id = _require_bounded_string(
        case.subject_id,
        label="subject id",
        max_bytes=active_policy.max_string_bytes,
    )
    selected_evidence_ids = _validated_ids(
        recommendation.evidence_ids,
        label="selected evidence id",
        max_bytes=active_policy.max_string_bytes,
    )
    selected_signal_ids = _validated_ids(
        recommendation.key_signal_ids,
        label="selected signal id",
        max_bytes=active_policy.max_string_bytes,
    )
    if len(set(selected_evidence_ids)) != len(selected_evidence_ids):
        _reject("duplicate_selected_evidence", "recommendation repeats an evidence id")
    if len(set(selected_signal_ids)) != len(selected_signal_ids):
        _reject("duplicate_selected_signal", "recommendation repeats a signal id")
    if len(selected_evidence_ids) > active_policy.max_evidence:
        _reject("evidence_budget_exceeded", "recommendation selects too much evidence")
    if len(selected_signal_ids) > active_policy.max_signals:
        _reject("signal_budget_exceeded", "recommendation selects too many signals")
    if (
        isinstance(recommendation.priority, bool)
        or not isinstance(recommendation.priority, int)
        or not 0 <= recommendation.priority <= 100
    ):
        _reject("invalid_priority", "recommendation priority must be an integer within [0, 100]")

    evidence_by_id = _unique_index(
        case.evidence,
        id_field="evidence_id",
        duplicate_code="duplicate_case_evidence",
        max_id_bytes=active_policy.max_string_bytes,
    )
    signals_by_id = _unique_index(
        case.signals,
        id_field="signal_id",
        duplicate_code="duplicate_case_signal",
        max_id_bytes=active_policy.max_string_bytes,
    )
    missing_evidence = [item for item in selected_evidence_ids if item not in evidence_by_id]
    if missing_evidence:
        _reject("missing_selected_evidence", "recommendation references missing evidence")
    missing_signals = [item for item in selected_signal_ids if item not in signals_by_id]
    if missing_signals:
        _reject("missing_selected_signal", "recommendation references a missing signal")

    evidence_aliases = {
        evidence_id: f"e{index}" for index, evidence_id in enumerate(selected_evidence_ids, start=1)
    }
    signal_aliases = {
        signal_id: f"s{index}" for index, signal_id in enumerate(selected_signal_ids, start=1)
    }

    projected_evidence: list[dict[str, Any]] = []
    excluded_attributes: list[tuple[str, int]] = []
    for evidence_id in selected_evidence_ids:
        ref = evidence_aliases[evidence_id]
        projected, excluded = _project_evidence(
            evidence_by_id[evidence_id],
            ref=ref,
            policy=active_policy,
        )
        projected_evidence.append(projected)
        excluded_attributes.append((ref, excluded))

    projected_signals = [
        _project_signal(
            signals_by_id[signal_id],
            ref=signal_aliases[signal_id],
            evidence_aliases=evidence_aliases,
            max_string_bytes=active_policy.max_string_bytes,
        )
        for signal_id in selected_signal_ids
    ]
    missing_information = [
        _require_bounded_string(
            item,
            label="missing information",
            max_bytes=active_policy.max_string_bytes,
        )
        for item in recommendation.missing_information
    ]
    uncertainty_notes = [
        _require_bounded_string(
            item,
            label="uncertainty note",
            max_bytes=active_policy.max_string_bytes,
        )
        for item in recommendation.uncertainty_notes
    ]

    context = {
        "schema_version": 1,
        "trust": "untrusted_case_evidence",
        "purpose": active_policy.purpose,
        "policy_id": active_policy.policy_id,
        "key_id": key_id,
        "case_ref": _pseudonym(
            secret=pseudonym_key,
            purpose=active_policy.purpose,
            label="case",
            value=case_id,
        ),
        "subject_ref": _pseudonym(
            secret=pseudonym_key,
            purpose=active_policy.purpose,
            label="subject",
            value=subject_id,
        ),
        "subject_type": _require_bounded_string(
            case.subject_type,
            label="subject type",
            max_bytes=active_policy.max_string_bytes,
        ),
        "recommendation": {
            "action": recommendation.recommended_action.value,
            "priority": recommendation.priority,
            "missing_information": missing_information,
            "uncertainty_notes": uncertainty_notes,
        },
        "evidence": projected_evidence,
        "signals": projected_signals,
    }
    canonical = json.dumps(
        context,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    encoded = canonical.encode("utf-8")
    if len(encoded) > active_policy.max_context_bytes:
        _reject(
            "context_size_exceeded",
            f"projected context is {len(encoded)} bytes; limit is {active_policy.max_context_bytes}",
        )
    digest = hashlib.sha256(encoded).hexdigest()
    manifest = ContextManifest(
        key_id=key_id,
        policy_id=active_policy.policy_id,
        case_id=case_id,
        context_sha256=digest,
        context_bytes=len(encoded),
        evidence_refs=tuple((alias, original) for original, alias in evidence_aliases.items()),
        signal_refs=tuple((alias, original) for original, alias in signal_aliases.items()),
        excluded_attributes=tuple(excluded_attributes),
    )
    return PromptContextBundle(canonical_model_json=canonical, manifest=manifest)
