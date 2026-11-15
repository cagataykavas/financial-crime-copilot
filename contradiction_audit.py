from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from copilot import EvidenceClaim, FinancialCrimeCase

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.:-]{0,127}\Z")


@dataclass(frozen=True)
class ClaimConflictPolicy:
    min_confidence: float = 0.80
    min_independent_sources: int = 2
    max_claims: int = 1_000
    max_text_bytes: int = 256
    max_findings: int = 100

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        if self.min_independent_sources < 2:
            raise ValueError("min_independent_sources must be at least two")
        if min(self.max_claims, self.max_text_bytes, self.max_findings) < 1:
            raise ValueError("claim audit budgets must be positive")


@dataclass(frozen=True)
class ClaimAuditIssue:
    code: str
    evidence_id: str | None = None
    claim_id: str | None = None


@dataclass(frozen=True)
class ClaimConflictFinding:
    group_sha256: str
    values_sha256: str
    evidence_ids: tuple[str, ...]
    source_count: int
    value_count: int


@dataclass(frozen=True)
class ClaimConflictReport:
    claims_seen: int
    claims_compared: int
    issues: tuple[ClaimAuditIssue, ...]
    findings: tuple[ClaimConflictFinding, ...]
    evidence_sha256: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded_text(value: object, *, limit: int) -> bool:
    return isinstance(value, str) and 0 < len(value.encode("utf-8")) <= limit


def _canonical_time(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC).isoformat()


def _canonical_value(claim: EvidenceClaim, *, limit: int) -> str | None:
    value = claim.value
    if type(value) not in {str, int, bool}:
        return None
    if isinstance(value, str) and not _bounded_text(value, limit=limit):
        return None
    return json.dumps(
        {"type": type(value).__name__, "value": value},
        sort_keys=True,
        separators=(",", ":"),
    )


def audit_claim_conflicts(
    case: FinancialCrimeCase,
    *,
    policy: ClaimConflictPolicy | None = None,
) -> ClaimConflictReport:
    """Find time-aligned, high-confidence claim conflicts without exposing claim values."""

    active_policy = policy or ClaimConflictPolicy()
    claims_seen = sum(len(evidence.claims) for evidence in case.evidence)
    if claims_seen > active_policy.max_claims:
        issue = ClaimAuditIssue("claim_budget_exceeded")
        return _report(claims_seen, 0, (issue,), (), active_policy)

    claim_id_counts = Counter(
        claim.claim_id for evidence in case.evidence for claim in evidence.claims
    )
    groups: dict[tuple[str, str, str], list[tuple[str, str, str]]] = defaultdict(list)
    issues: list[ClaimAuditIssue] = []
    claims_compared = 0

    for evidence in case.evidence:
        source_valid = _bounded_text(evidence.source, limit=active_policy.max_text_bytes)
        for claim in evidence.claims:
            issue_codes: list[str] = []
            if not _bounded_text(claim.claim_id, limit=active_policy.max_text_bytes):
                issue_codes.append("invalid_claim_id")
            elif claim_id_counts[claim.claim_id] > 1:
                issue_codes.append("duplicate_claim_id")
            if not _bounded_text(claim.subject_id, limit=active_policy.max_text_bytes):
                issue_codes.append("invalid_claim_subject")
            if not isinstance(claim.predicate, str) or not _IDENTIFIER.fullmatch(claim.predicate):
                issue_codes.append("invalid_claim_predicate")
            effective_at = _canonical_time(claim.effective_at)
            if effective_at is None:
                issue_codes.append("invalid_claim_effective_at")
            value = _canonical_value(claim, limit=active_policy.max_text_bytes)
            if value is None:
                issue_codes.append("invalid_claim_value")
            if not source_valid:
                issue_codes.append("invalid_evidence_source")

            for code in dict.fromkeys(issue_codes):
                issues.append(ClaimAuditIssue(code, evidence.evidence_id, claim.claim_id))
            if issue_codes or evidence.confidence < active_policy.min_confidence:
                continue

            claims_compared += 1
            key = (claim.subject_id, claim.predicate, effective_at)
            groups[key].append((evidence.source, evidence.evidence_id, value))

    findings: list[ClaimConflictFinding] = []
    for key, rows in sorted(groups.items()):
        sources = {source for source, _, _ in rows}
        values = {value for _, _, value in rows}
        if len(sources) < active_policy.min_independent_sources or len(values) < 2:
            continue
        group_payload = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
        values_payload = json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":"))
        findings.append(
            ClaimConflictFinding(
                group_sha256=_sha256(group_payload),
                values_sha256=_sha256(values_payload),
                evidence_ids=tuple(sorted({evidence_id for _, evidence_id, _ in rows})),
                source_count=len(sources),
                value_count=len(values),
            )
        )

    if len(findings) > active_policy.max_findings:
        issues.append(ClaimAuditIssue("conflict_finding_budget_exceeded"))
        findings = findings[: active_policy.max_findings]
    return _report(
        claims_seen,
        claims_compared,
        tuple(issues),
        tuple(findings),
        active_policy,
    )


def _report(
    claims_seen: int,
    claims_compared: int,
    issues: tuple[ClaimAuditIssue, ...],
    findings: tuple[ClaimConflictFinding, ...],
    policy: ClaimConflictPolicy,
) -> ClaimConflictReport:
    payload = {
        "claims_seen": claims_seen,
        "claims_compared": claims_compared,
        "issues": [asdict(issue) for issue in issues],
        "findings": [asdict(finding) for finding in findings],
        "policy": asdict(policy),
    }
    evidence_sha256 = _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return ClaimConflictReport(
        claims_seen=claims_seen,
        claims_compared=claims_compared,
        issues=issues,
        findings=findings,
        evidence_sha256=evidence_sha256,
    )
