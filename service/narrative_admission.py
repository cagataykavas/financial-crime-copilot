from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from copilot import CopilotRecommendation, FinancialCrimeCase

SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class NarrativeAdmissionError(ValueError):
    """A stable, fail-closed rejection at the model-output boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True)
class NarrativePolicy:
    policy_id: str = "grounded-narrative-v1"
    max_payload_bytes: int = 16_384
    max_claims: int = 12
    max_claim_bytes: int = 600
    max_total_claim_bytes: int = 4_000
    max_references_per_claim: int = 12

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", self.policy_id):
            raise ValueError("policy_id must be a safe identifier")
        limits = {
            "max_payload_bytes": (self.max_payload_bytes, 1_048_576),
            "max_claims": (self.max_claims, 100),
            "max_claim_bytes": (self.max_claim_bytes, 16_384),
            "max_total_claim_bytes": (self.max_total_claim_bytes, 65_536),
            "max_references_per_claim": (self.max_references_per_claim, 100),
        }
        for name, (value, maximum) in limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")


@dataclass(frozen=True)
class AdmittedClaim:
    claim_id: str
    text: str
    evidence_ids: tuple[str, ...]
    signal_ids: tuple[str, ...]


@dataclass(frozen=True)
class NarrativeReceipt:
    schema_version: int
    policy_id: str
    case_ref: str
    context_sha256: str
    narrative_sha256: str
    binding_sha256: str
    claim_count: int
    evidence_references: int
    signal_references: int


@dataclass(frozen=True)
class NarrativeAdmission:
    claims: tuple[AdmittedClaim, ...]
    receipt: NarrativeReceipt


def _reject(code: str, detail: str) -> None:
    raise NarrativeAdmissionError(code, detail)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_constant(_value: str) -> None:
    _reject("invalid_json", "non-finite JSON numbers are forbidden")


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _reject("duplicate_field", "model output contains a duplicate object field")
        result[key] = value
    return result


def _load_payload(raw: str | bytes, maximum: int) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            encoded = raw.encode("utf-8")
        except UnicodeError:
            _reject("invalid_encoding", "model output is not valid UTF-8")
    elif isinstance(raw, bytes):
        if len(raw) > maximum:
            _reject("payload_budget_exceeded", "model output exceeds the byte budget")
        encoded = raw
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            _reject("invalid_encoding", "model output is not valid UTF-8")
    else:
        _reject("invalid_payload_type", "model output must be UTF-8 text or bytes")
    if len(encoded) > maximum:
        _reject("payload_budget_exceeded", "model output exceeds the byte budget")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
        )
    except NarrativeAdmissionError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError):
        _reject("invalid_json", "model output is not valid strict JSON")
    if not isinstance(value, dict):
        _reject("invalid_shape", "model output must be an object")
    return value


def _exact_fields(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _reject("invalid_shape", f"{label} must be an object")
    if set(value) != expected:
        _reject("invalid_fields", f"{label} fields do not match the contract")
    return value


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        _reject("invalid_identifier", f"{label} must be a safe bounded identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        _reject("invalid_digest", f"{label} must be a lowercase SHA-256 digest")
    return value


def _references(value: Any, label: str, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum:
        _reject("reference_budget_exceeded", f"{label} exceeds its reference budget")
    result = tuple(_safe_id(item, label) for item in value)
    if len(result) != len(set(result)):
        _reject("duplicate_reference", f"{label} contains a duplicate reference")
    return result


def _claim_text(value: Any, policy: NarrativePolicy) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _reject("invalid_claim_text", "claim text must be a non-empty trimmed string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        _reject("invalid_claim_text", "claim text must be valid UTF-8")
    if len(encoded) > policy.max_claim_bytes:
        _reject("claim_budget_exceeded", "claim text exceeds the byte budget")
    if any(unicodedata.category(character).startswith("C") for character in value):
        _reject("unsafe_claim_text", "claim text contains control or format characters")
    return value


def admit_narrative(
    raw: str | bytes,
    case: FinancialCrimeCase,
    recommendation: CopilotRecommendation,
    *,
    expected_context_sha256: str,
    policy: NarrativePolicy | None = None,
) -> NarrativeAdmission:
    """Validate a model-written narrative against its deterministic evidence boundary.

    Acceptance proves structural binding and citation coverage. It does not infer
    whether a claim is entailed by the referenced evidence.
    """

    active_policy = policy or NarrativePolicy()
    expected_context = _sha256(expected_context_sha256, "expected context")
    payload = _exact_fields(
        _load_payload(raw, active_policy.max_payload_bytes),
        {"schema_version", "case_id", "action", "context_sha256", "claims"},
        "narrative",
    )
    if isinstance(payload["schema_version"], bool) or payload["schema_version"] != SCHEMA_VERSION:
        _reject("unsupported_schema", "narrative schema version is unsupported")
    case_id = _safe_id(case.case_id, "case id")
    if recommendation.case_id != case_id:
        _reject("recommendation_case_mismatch", "recommendation does not belong to the case")
    if payload["case_id"] != case_id:
        _reject("case_mismatch", "narrative does not belong to the case")
    if payload["action"] != recommendation.recommended_action.value:
        _reject("action_mismatch", "narrative action does not match the recommendation")
    if _sha256(payload["context_sha256"], "narrative context") != expected_context:
        _reject("context_mismatch", "narrative is bound to a different model context")

    selected_evidence = tuple(
        _safe_id(item, "recommendation evidence") for item in recommendation.evidence_ids
    )
    selected_signals = tuple(
        _safe_id(item, "recommendation signal") for item in recommendation.key_signal_ids
    )
    if len(selected_evidence) != len(set(selected_evidence)):
        _reject("duplicate_recommendation_evidence", "recommendation repeats evidence")
    if len(selected_signals) != len(set(selected_signals)):
        _reject("duplicate_recommendation_signal", "recommendation repeats a signal")
    case_evidence = tuple(_safe_id(item.evidence_id, "case evidence") for item in case.evidence)
    case_signals = tuple(_safe_id(item.signal_id, "case signal") for item in case.signals)
    if len(case_evidence) != len(set(case_evidence)):
        _reject("duplicate_case_evidence", "case contains ambiguous evidence identifiers")
    if len(case_signals) != len(set(case_signals)):
        _reject("duplicate_case_signal", "case contains ambiguous signal identifiers")
    known_evidence = set(case_evidence)
    known_signals = set(case_signals)
    if not selected_evidence:
        _reject("ungrounded_recommendation", "recommendation contains no evidence")
    if not set(selected_evidence) <= known_evidence:
        _reject("recommendation_evidence_missing", "recommendation cites missing evidence")
    if not set(selected_signals) <= known_signals:
        _reject("recommendation_signal_missing", "recommendation cites a missing signal")

    claims = payload["claims"]
    if not isinstance(claims, list) or not 1 <= len(claims) <= active_policy.max_claims:
        _reject("claim_budget_exceeded", "narrative must contain a bounded non-empty claim list")
    claim_ids: set[str] = set()
    covered_evidence: set[str] = set()
    covered_signals: set[str] = set()
    admitted_claims: list[AdmittedClaim] = []
    total_claim_bytes = 0
    for claim_value in claims:
        claim = _exact_fields(
            claim_value,
            {"claim_id", "text", "evidence_ids", "signal_ids"},
            "claim",
        )
        claim_id = _safe_id(claim["claim_id"], "claim id")
        if claim_id in claim_ids:
            _reject("duplicate_claim", "claim id appears more than once")
        claim_ids.add(claim_id)
        text = _claim_text(claim["text"], active_policy)
        total_claim_bytes += len(text.encode("utf-8"))
        evidence = _references(
            claim["evidence_ids"],
            "claim evidence",
            active_policy.max_references_per_claim,
        )
        signals = _references(
            claim["signal_ids"],
            "claim signals",
            active_policy.max_references_per_claim,
        )
        if not evidence:
            _reject("uncited_claim", "every claim must cite selected evidence")
        if not set(evidence) <= set(selected_evidence):
            _reject("unknown_evidence_reference", "claim cites evidence outside the recommendation")
        if not set(signals) <= set(selected_signals):
            _reject("unknown_signal_reference", "claim cites a signal outside the recommendation")
        covered_evidence.update(evidence)
        covered_signals.update(signals)
        admitted_claims.append(
            AdmittedClaim(
                claim_id=claim_id,
                text=text,
                evidence_ids=evidence,
                signal_ids=signals,
            )
        )
    if total_claim_bytes > active_policy.max_total_claim_bytes:
        _reject("narrative_budget_exceeded", "total claim text exceeds the byte budget")
    if covered_evidence != set(selected_evidence):
        _reject("incomplete_evidence_coverage", "narrative omits selected evidence")
    if covered_signals != set(selected_signals):
        _reject("incomplete_signal_coverage", "narrative omits selected signals")

    narrative_sha256 = _digest(_canonical(payload))
    binding = {
        "case_id": case_id,
        "context_sha256": expected_context,
        "evidence_ids": selected_evidence,
        "narrative_sha256": narrative_sha256,
        "policy_id": active_policy.policy_id,
        "recommended_action": recommendation.recommended_action.value,
        "signal_ids": selected_signals,
    }
    return NarrativeAdmission(
        claims=tuple(admitted_claims),
        receipt=NarrativeReceipt(
            schema_version=SCHEMA_VERSION,
            policy_id=active_policy.policy_id,
            case_ref=_digest(case_id.encode("utf-8"))[:24],
            context_sha256=expected_context,
            narrative_sha256=narrative_sha256,
            binding_sha256=_digest(_canonical(binding)),
            claim_count=len(claims),
            evidence_references=len(covered_evidence),
            signal_references=len(covered_signals),
        ),
    )
