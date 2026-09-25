"""Fail-closed admission for results returned to an agent by external tools.

The gate binds an observed result to a previously prepared call intent.  It is
deliberately independent from the tool transport so a caller can place it at
the last boundary before tool output enters an agent's context.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime


class ArtifactError(ValueError):
    """The supplied artifact cannot be represented safely as bounded JSON."""


@dataclass(frozen=True)
class AdmissionPolicy:
    policy_id: str
    max_document_bytes: int = 131_072
    max_json_depth: int = 12
    max_json_nodes: int = 4_096
    max_intent_ttl_seconds: float = 300.0
    max_execution_seconds: float = 60.0
    max_result_age_seconds: float = 30.0
    max_future_skew_seconds: float = 5.0
    max_attempt: int = 3

    def __post_init__(self) -> None:
        _require_identifier(self.policy_id, "policy_id")
        for name in ("max_document_bytes", "max_json_depth", "max_json_nodes", "max_attempt"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "max_intent_ttl_seconds",
            "max_execution_seconds",
            "max_result_age_seconds",
            "max_future_skew_seconds",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a finite non-negative number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True)
class ToolCallIntent:
    run_id: str
    call_id: str
    tool_name: str
    arguments_digest: str
    policy_id: str
    allowed_producers: tuple[str, ...]
    issued_at: str
    expires_at: str


@dataclass(frozen=True)
class ToolResultReceipt:
    run_id: str
    call_id: str
    tool_name: str
    arguments_digest: str
    result_digest: str
    producer_id: str
    attempt: int
    status: str
    started_at: str
    completed_at: str


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    reason_codes: tuple[str, ...]
    intent_digest: str
    receipt_digest: str
    result_digest: str
    call_binding_digest: str
    evidence_digest: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 128:
        raise ArtifactError(f"{name} must be a non-empty UTF-8 string of at most 128 bytes")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ArtifactError(f"{name} contains a control character")
    return value


def _bounded_json(value: object, policy: AdmissionPolicy) -> bytes:
    nodes = 0
    active: set[int] = set()

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > policy.max_json_nodes:
            raise ArtifactError("JSON node budget exceeded")
        if depth > policy.max_json_depth:
            raise ArtifactError("JSON depth budget exceeded")

        if item is None or isinstance(item, (str, bool)):
            return
        if isinstance(item, int):
            if abs(item) > 2**63 - 1:
                raise ArtifactError("integer is outside signed 64-bit range")
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ArtifactError("non-finite number is not valid evidence")
            return
        if not isinstance(item, (dict, list, tuple)):
            raise ArtifactError(f"unsupported JSON value: {type(item).__name__}")

        object_id = id(item)
        if object_id in active:
            raise ArtifactError("cyclic JSON value")
        active.add(object_id)
        try:
            if isinstance(item, dict):
                keys = tuple(item)
                for key in keys:
                    if not isinstance(key, str):
                        raise ArtifactError("JSON object keys must be strings")
                for key in sorted(keys):
                    visit(key, depth + 1)
                    visit(item[key], depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
        finally:
            active.remove(object_id)

    visit(value, 0)
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactError("value is not canonical JSON") from exc
    if len(encoded) > policy.max_document_bytes:
        raise ArtifactError("JSON byte budget exceeded")
    return encoded


def _digest_json(value: object, policy: AdmissionPolicy) -> str:
    return hashlib.sha256(_bounded_json(value, policy)).hexdigest()


def _parse_time(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ArtifactError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ArtifactError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ArtifactError(f"{name} must include a timezone")
    return parsed.astimezone(UTC)


def _iso_utc(value: datetime, name: str) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ArtifactError(f"{name} must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def prepare_tool_call(
    *,
    run_id: str,
    call_id: str,
    tool_name: str,
    arguments: object,
    policy: AdmissionPolicy,
    allowed_producers: tuple[str, ...],
    issued_at: datetime,
    expires_at: datetime,
) -> ToolCallIntent:
    """Create the content-addressed intent that must precede tool execution."""
    run_id = _require_identifier(run_id, "run_id")
    call_id = _require_identifier(call_id, "call_id")
    tool_name = _require_identifier(tool_name, "tool_name")
    producers = tuple(
        sorted({_require_identifier(item, "producer_id") for item in allowed_producers})
    )
    if not producers:
        raise ArtifactError("allowed_producers must not be empty")
    issued = _iso_utc(issued_at, "issued_at")
    expires = _iso_utc(expires_at, "expires_at")
    if _parse_time(expires, "expires_at") <= _parse_time(issued, "issued_at"):
        raise ArtifactError("expires_at must be after issued_at")
    return ToolCallIntent(
        run_id=run_id,
        call_id=call_id,
        tool_name=tool_name,
        arguments_digest=_digest_json(arguments, policy),
        policy_id=policy.policy_id,
        allowed_producers=producers,
        issued_at=issued,
        expires_at=expires,
    )


def issue_result_receipt(
    intent: ToolCallIntent,
    *,
    result: object,
    producer_id: str,
    attempt: int,
    status: str,
    started_at: datetime,
    completed_at: datetime,
    policy: AdmissionPolicy,
) -> ToolResultReceipt:
    """Create a receipt as a conforming tool adapter would after execution."""
    return ToolResultReceipt(
        run_id=intent.run_id,
        call_id=intent.call_id,
        tool_name=intent.tool_name,
        arguments_digest=intent.arguments_digest,
        result_digest=_digest_json(result, policy),
        producer_id=_require_identifier(producer_id, "producer_id"),
        attempt=attempt,
        status=status,
        started_at=_iso_utc(started_at, "started_at"),
        completed_at=_iso_utc(completed_at, "completed_at"),
    )


_REASON_ORDER = (
    "INVALID_INTENT",
    "INVALID_RECEIPT",
    "INVALID_RESULT_PAYLOAD",
    "POLICY_MISMATCH",
    "RUN_MISMATCH",
    "CALL_MISMATCH",
    "TOOL_MISMATCH",
    "ARGUMENTS_MISMATCH",
    "UNAUTHORIZED_PRODUCER",
    "UNSUCCESSFUL_RESULT",
    "ATTEMPT_OUT_OF_POLICY",
    "INTENT_TTL_OUT_OF_POLICY",
    "INTENT_NOT_YET_VALID",
    "INTENT_EXPIRED",
    "RECEIPT_BEFORE_INTENT",
    "RECEIPT_TIME_REVERSED",
    "EXECUTION_TIMEOUT",
    "RESULT_FROM_FUTURE",
    "RESULT_STALE",
    "RESULT_DIGEST_MISMATCH",
    "RECEIPT_REPLAYED",
    "CALL_ALREADY_ADMITTED",
)
_REASON_RANK = {code: index for index, code in enumerate(_REASON_ORDER)}


class ToolResultAdmissionLedger:
    """Atomically admit each bound result at most once within this process."""

    def __init__(self, policy: AdmissionPolicy) -> None:
        self.policy = policy
        self._lock = threading.Lock()
        self._admitted_calls: set[tuple[str, str]] = set()
        self._admitted_receipts: set[str] = set()

    def _decision(
        self,
        *,
        reasons: set[str],
        intent_digest: str,
        receipt_digest: str,
        result_digest: str,
        call_binding_digest: str,
    ) -> AdmissionDecision:
        reason_codes = tuple(sorted(reasons, key=_REASON_RANK.__getitem__))
        report = {
            "accepted": not reason_codes,
            "reason_codes": reason_codes,
            "intent_digest": intent_digest,
            "receipt_digest": receipt_digest,
            "result_digest": result_digest,
            "call_binding_digest": call_binding_digest,
        }
        evidence_digest = _digest_json(report, self.policy)
        return AdmissionDecision(evidence_digest=evidence_digest, **report)

    def admit(
        self,
        intent: ToolCallIntent,
        receipt: ToolResultReceipt,
        *,
        result: object,
        observed_at: datetime,
    ) -> AdmissionDecision:
        """Validate a tool result and consume its call binding exactly once."""
        reasons: set[str] = set()
        unavailable = "unavailable"
        intent_digest = receipt_digest = result_digest = call_binding_digest = unavailable

        try:
            intent_payload = asdict(intent)
            for name in ("run_id", "call_id", "tool_name", "policy_id"):
                _require_identifier(intent_payload[name], name)
            if not isinstance(intent.allowed_producers, tuple) or not intent.allowed_producers:
                raise ArtifactError("allowed_producers must be a non-empty tuple")
            for producer in intent.allowed_producers:
                _require_identifier(producer, "producer_id")
            intent_digest = _digest_json(intent_payload, self.policy)
            issued = _parse_time(intent.issued_at, "issued_at")
            expires = _parse_time(intent.expires_at, "expires_at")
        except (ArtifactError, TypeError, KeyError):
            reasons.add("INVALID_INTENT")
            issued = expires = None

        try:
            receipt_payload = asdict(receipt)
            for name in ("run_id", "call_id", "tool_name", "producer_id"):
                _require_identifier(receipt_payload[name], name)
            receipt_digest = _digest_json(receipt_payload, self.policy)
            started = _parse_time(receipt.started_at, "started_at")
            completed = _parse_time(receipt.completed_at, "completed_at")
        except (ArtifactError, TypeError, KeyError):
            reasons.add("INVALID_RECEIPT")
            started = completed = None

        try:
            result_digest = _digest_json(result, self.policy)
        except ArtifactError:
            reasons.add("INVALID_RESULT_PAYLOAD")

        try:
            if observed_at.tzinfo is None or observed_at.utcoffset() is None:
                raise ValueError
            observed = observed_at.astimezone(UTC)
        except (AttributeError, ValueError):
            reasons.add("INVALID_RECEIPT")
            observed = None

        if "INVALID_INTENT" not in reasons:
            call_binding_digest = _digest_json(
                {"run_id": intent.run_id, "call_id": intent.call_id}, self.policy
            )
            if intent.policy_id != self.policy.policy_id:
                reasons.add("POLICY_MISMATCH")

        if "INVALID_INTENT" not in reasons and "INVALID_RECEIPT" not in reasons:
            if receipt.run_id != intent.run_id:
                reasons.add("RUN_MISMATCH")
            if receipt.call_id != intent.call_id:
                reasons.add("CALL_MISMATCH")
            if receipt.tool_name != intent.tool_name:
                reasons.add("TOOL_MISMATCH")
            if receipt.arguments_digest != intent.arguments_digest:
                reasons.add("ARGUMENTS_MISMATCH")
            if receipt.producer_id not in intent.allowed_producers:
                reasons.add("UNAUTHORIZED_PRODUCER")
            if receipt.status != "success":
                reasons.add("UNSUCCESSFUL_RESULT")
            if (
                isinstance(receipt.attempt, bool)
                or not isinstance(receipt.attempt, int)
                or not 1 <= receipt.attempt <= self.policy.max_attempt
            ):
                reasons.add("ATTEMPT_OUT_OF_POLICY")
            if result_digest != unavailable and receipt.result_digest != result_digest:
                reasons.add("RESULT_DIGEST_MISMATCH")

        if issued is not None and expires is not None and observed is not None:
            skew = self.policy.max_future_skew_seconds
            if (expires - issued).total_seconds() > self.policy.max_intent_ttl_seconds:
                reasons.add("INTENT_TTL_OUT_OF_POLICY")
            if observed.timestamp() + skew < issued.timestamp():
                reasons.add("INTENT_NOT_YET_VALID")
            if observed.timestamp() - skew > expires.timestamp():
                reasons.add("INTENT_EXPIRED")

        if None not in (issued, started, completed, observed):
            assert issued is not None and started is not None
            assert completed is not None and observed is not None
            skew = self.policy.max_future_skew_seconds
            if started.timestamp() + skew < issued.timestamp():
                reasons.add("RECEIPT_BEFORE_INTENT")
            if completed < started:
                reasons.add("RECEIPT_TIME_REVERSED")
            if (completed - started).total_seconds() > self.policy.max_execution_seconds:
                reasons.add("EXECUTION_TIMEOUT")
            if completed.timestamp() > observed.timestamp() + skew:
                reasons.add("RESULT_FROM_FUTURE")
            if (observed - completed).total_seconds() > self.policy.max_result_age_seconds:
                reasons.add("RESULT_STALE")

        if reasons:
            return self._decision(
                reasons=reasons,
                intent_digest=intent_digest,
                receipt_digest=receipt_digest,
                result_digest=result_digest,
                call_binding_digest=call_binding_digest,
            )

        call_key = (intent.run_id, intent.call_id)
        with self._lock:
            if receipt_digest in self._admitted_receipts:
                reasons.add("RECEIPT_REPLAYED")
            if call_key in self._admitted_calls:
                reasons.add("CALL_ALREADY_ADMITTED")
            if not reasons:
                self._admitted_receipts.add(receipt_digest)
                self._admitted_calls.add(call_key)

        return self._decision(
            reasons=reasons,
            intent_digest=intent_digest,
            receipt_digest=receipt_digest,
            result_digest=result_digest,
            call_binding_digest=call_binding_digest,
        )
