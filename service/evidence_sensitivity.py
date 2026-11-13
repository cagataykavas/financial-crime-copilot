from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PASS_EXIT_CODE = 0
POLICY_FAILURE_EXIT_CODE = 2
INVALID_EVIDENCE_EXIT_CODE = 3
MAX_INPUT_BYTES = 8_000_000
MAX_TRIALS_LIMIT = 10_000
MAX_CLAIMS_PER_TRIAL_LIMIT = 1_000
MAX_EVIDENCE_PER_CLAIM_LIMIT = 100
MAX_EVIDENCE_INVENTORY_LIMIT = 10_000
MAX_TOTAL_CLAIMS = 100_000
MAX_REPORTED_FINDINGS = 100


class EvidenceSensitivityError(ValueError):
    """Counterfactual evaluation evidence is malformed or exceeds a bound."""


def _integer(name: str, value: int, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceSensitivityError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise EvidenceSensitivityError(f"{name} must be between {minimum} and {maximum}")


def _rate(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceSensitivityError(f"{name} must be numeric")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise EvidenceSensitivityError(f"{name} must be finite and between 0 and 1")


def _identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise EvidenceSensitivityError(f"{name} must be a non-empty string")
    if value != value.strip() or len(value) > 128:
        raise EvidenceSensitivityError(f"{name} must be trimmed and at most 128 characters")
    if any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise EvidenceSensitivityError(f"{name} must contain printable ASCII without spaces")


@dataclass(frozen=True, slots=True)
class ClaimTrace:
    claim_id: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier("claim_id", self.claim_id)
        if not isinstance(self.evidence_ids, tuple) or not self.evidence_ids:
            raise EvidenceSensitivityError("evidence_ids must be a non-empty tuple")
        if len(self.evidence_ids) > MAX_EVIDENCE_PER_CLAIM_LIMIT:
            raise EvidenceSensitivityError("evidence_ids exceeds the hard limit")
        for evidence_id in self.evidence_ids:
            _identifier("evidence_id", evidence_id)
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise EvidenceSensitivityError("evidence_ids must be unique within a claim")


@dataclass(frozen=True, slots=True)
class EvidenceAblationTrial:
    trial_id: str
    ablated_evidence_id: str
    retained_evidence_ids: tuple[str, ...]
    baseline_claims: tuple[ClaimTrace, ...]
    counterfactual_claims: tuple[ClaimTrace, ...]

    def __post_init__(self) -> None:
        _identifier("trial_id", self.trial_id)
        _identifier("ablated_evidence_id", self.ablated_evidence_id)
        if not isinstance(self.retained_evidence_ids, tuple):
            raise EvidenceSensitivityError("retained_evidence_ids must be a tuple")
        if len(self.retained_evidence_ids) > MAX_EVIDENCE_INVENTORY_LIMIT:
            raise EvidenceSensitivityError("retained evidence inventory exceeds the hard limit")
        for evidence_id in self.retained_evidence_ids:
            _identifier("retained evidence_id", evidence_id)
        if len(self.retained_evidence_ids) != len(set(self.retained_evidence_ids)):
            raise EvidenceSensitivityError("retained_evidence_ids must be unique")
        if self.ablated_evidence_id in self.retained_evidence_ids:
            raise EvidenceSensitivityError("ablated evidence must not remain in the inventory")
        for name, claims in (
            ("baseline_claims", self.baseline_claims),
            ("counterfactual_claims", self.counterfactual_claims),
        ):
            if not isinstance(claims, tuple):
                raise EvidenceSensitivityError(f"{name} must be a tuple")
            if len(claims) > MAX_CLAIMS_PER_TRIAL_LIMIT:
                raise EvidenceSensitivityError(f"{name} exceeds the hard limit")
            if not all(isinstance(claim, ClaimTrace) for claim in claims):
                raise EvidenceSensitivityError(f"{name} must contain ClaimTrace values")
            claim_ids = [claim.claim_id for claim in claims]
            if len(claim_ids) != len(set(claim_ids)):
                raise EvidenceSensitivityError(f"{name} contains duplicate claim IDs")
        baseline_inventory = set(self.retained_evidence_ids) | {self.ablated_evidence_id}
        if any(not set(claim.evidence_ids) <= baseline_inventory for claim in self.baseline_claims):
            raise EvidenceSensitivityError(
                "baseline claim cites evidence outside the declared inventory"
            )


@dataclass(frozen=True, slots=True)
class EvidenceSensitivityPolicy:
    min_trials: int = 20
    min_affected_claims: int = 20
    min_unique_support_claims: int = 10
    min_unaffected_claims: int = 20
    min_unique_withdrawal_rate: float = 0.95
    min_affected_adjustment_rate: float = 1.0
    min_unaffected_retention_rate: float = 0.90
    max_irrelevant_ablation_trials: int = 0
    max_unexpected_claims: int = 0
    max_trials: int = 2_000
    max_claims_per_trial: int = 100
    max_evidence_per_claim: int = 16
    max_evidence_inventory: int = 200

    def __post_init__(self) -> None:
        _integer("min_trials", self.min_trials, minimum=1, maximum=MAX_TRIALS_LIMIT)
        _integer(
            "min_affected_claims",
            self.min_affected_claims,
            minimum=1,
            maximum=MAX_TOTAL_CLAIMS,
        )
        _integer(
            "min_unique_support_claims",
            self.min_unique_support_claims,
            minimum=1,
            maximum=MAX_TOTAL_CLAIMS,
        )
        _integer(
            "min_unaffected_claims",
            self.min_unaffected_claims,
            minimum=1,
            maximum=MAX_TOTAL_CLAIMS,
        )
        _integer(
            "max_irrelevant_ablation_trials",
            self.max_irrelevant_ablation_trials,
            minimum=0,
            maximum=MAX_TRIALS_LIMIT,
        )
        _integer(
            "max_unexpected_claims",
            self.max_unexpected_claims,
            minimum=0,
            maximum=MAX_TOTAL_CLAIMS,
        )
        _integer(
            "max_trials",
            self.max_trials,
            minimum=self.min_trials,
            maximum=MAX_TRIALS_LIMIT,
        )
        _integer(
            "max_claims_per_trial",
            self.max_claims_per_trial,
            minimum=1,
            maximum=MAX_CLAIMS_PER_TRIAL_LIMIT,
        )
        _integer(
            "max_evidence_per_claim",
            self.max_evidence_per_claim,
            minimum=1,
            maximum=MAX_EVIDENCE_PER_CLAIM_LIMIT,
        )
        _integer(
            "max_evidence_inventory",
            self.max_evidence_inventory,
            minimum=1,
            maximum=MAX_EVIDENCE_INVENTORY_LIMIT,
        )
        _rate("min_unique_withdrawal_rate", self.min_unique_withdrawal_rate)
        _rate("min_affected_adjustment_rate", self.min_affected_adjustment_rate)
        _rate("min_unaffected_retention_rate", self.min_unaffected_retention_rate)


@dataclass(frozen=True, slots=True)
class SensitivityFinding:
    code: str
    trial_ref: str
    claim_ref: str | None = None


@dataclass(frozen=True, slots=True)
class SensitivityViolation:
    code: str


@dataclass(frozen=True, slots=True)
class EvidenceSensitivityReport:
    passed: bool
    trials: int
    affected_claims: int
    unique_support_claims: int
    unaffected_claims: int
    irrelevant_ablation_trials: int
    unexpected_claims: int
    counterfactual_citation_failures: int
    unique_withdrawal_rate: float | None
    affected_adjustment_rate: float | None
    unaffected_retention_rate: float | None
    findings: tuple[SensitivityFinding, ...]
    findings_truncated: bool
    violations: tuple[SensitivityViolation, ...]
    evidence_digest: str
    policy: EvidenceSensitivityPolicy

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = "counterfactual-evidence-sensitivity-v1"
        return payload


def _reference(domain: str, value: str) -> str:
    return hashlib.sha256(f"{domain}\0{value}".encode()).hexdigest()[:16]


def _claim_payload(claim: ClaimTrace) -> dict[str, Any]:
    return {
        "claim_id": claim.claim_id,
        "evidence_ids": sorted(claim.evidence_ids),
    }


def _digest(trials: Sequence[EvidenceAblationTrial], policy: EvidenceSensitivityPolicy) -> str:
    payload = {
        "policy": asdict(policy),
        "schema_version": "counterfactual-evidence-evaluation-v1",
        "trials": [
            {
                "ablated_evidence_id": trial.ablated_evidence_id,
                "baseline_claims": [
                    _claim_payload(claim)
                    for claim in sorted(trial.baseline_claims, key=lambda item: item.claim_id)
                ],
                "counterfactual_claims": [
                    _claim_payload(claim)
                    for claim in sorted(trial.counterfactual_claims, key=lambda item: item.claim_id)
                ],
                "retained_evidence_ids": sorted(trial.retained_evidence_ids),
                "trial_id": trial.trial_id,
            }
            for trial in sorted(trials, key=lambda item: item.trial_id)
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def audit_evidence_sensitivity(
    trials: Sequence[EvidenceAblationTrial],
    *,
    policy: EvidenceSensitivityPolicy | None = None,
) -> EvidenceSensitivityReport:
    active_policy = policy or EvidenceSensitivityPolicy()
    rows = tuple(trials)
    if not 1 <= len(rows) <= active_policy.max_trials:
        raise EvidenceSensitivityError(
            f"trial count must be between 1 and {active_policy.max_trials}"
        )
    if not all(isinstance(trial, EvidenceAblationTrial) for trial in rows):
        raise EvidenceSensitivityError("trials must contain EvidenceAblationTrial values")
    trial_ids = [trial.trial_id for trial in rows]
    if len(trial_ids) != len(set(trial_ids)):
        raise EvidenceSensitivityError("trial_id values must be unique")

    total_claims = sum(
        len(trial.baseline_claims) + len(trial.counterfactual_claims) for trial in rows
    )
    if total_claims > MAX_TOTAL_CLAIMS:
        raise EvidenceSensitivityError(
            f"evaluation exceeds the {MAX_TOTAL_CLAIMS} claim work limit"
        )
    for trial in rows:
        if max(len(trial.baseline_claims), len(trial.counterfactual_claims)) > (
            active_policy.max_claims_per_trial
        ):
            raise EvidenceSensitivityError("trial exceeds the policy claim limit")
        if len(trial.retained_evidence_ids) > active_policy.max_evidence_inventory:
            raise EvidenceSensitivityError("trial exceeds the policy evidence inventory limit")
        if any(
            len(claim.evidence_ids) > active_policy.max_evidence_per_claim
            for claim in trial.baseline_claims + trial.counterfactual_claims
        ):
            raise EvidenceSensitivityError("claim exceeds the policy citation limit")

    affected = 0
    adjusted = 0
    unique = 0
    unique_withdrawn = 0
    unaffected = 0
    unaffected_retained = 0
    irrelevant_trials = 0
    unexpected = 0
    citation_failures = 0
    all_findings: list[SensitivityFinding] = []

    def finding(code: str, trial_id: str, claim_id: str | None = None) -> None:
        all_findings.append(
            SensitivityFinding(
                code=code,
                trial_ref=_reference("trial", trial_id),
                claim_ref=(
                    _reference("claim", f"{trial_id}\0{claim_id}") if claim_id is not None else None
                ),
            )
        )

    for trial in sorted(rows, key=lambda item: item.trial_id):
        baseline_by_id = {claim.claim_id: claim for claim in trial.baseline_claims}
        counterfactual_by_id = {claim.claim_id: claim for claim in trial.counterfactual_claims}
        retained_inventory = set(trial.retained_evidence_ids)
        affected_ids = {
            claim.claim_id
            for claim in trial.baseline_claims
            if trial.ablated_evidence_id in claim.evidence_ids
        }
        if not affected_ids:
            irrelevant_trials += 1
            finding("irrelevant_ablation", trial.trial_id)

        unexpected_ids = sorted(set(counterfactual_by_id) - set(baseline_by_id))
        unexpected += len(unexpected_ids)
        for claim_id in unexpected_ids:
            finding("unexpected_claim", trial.trial_id, claim_id)

        valid_counterfactual: dict[str, bool] = {}
        for claim in trial.counterfactual_claims:
            citations = set(claim.evidence_ids)
            is_valid = True
            if trial.ablated_evidence_id in citations:
                finding(
                    "counterfactual_cites_ablated_evidence",
                    trial.trial_id,
                    claim.claim_id,
                )
                is_valid = False
            if not citations <= retained_inventory:
                finding(
                    "counterfactual_cites_unknown_evidence",
                    trial.trial_id,
                    claim.claim_id,
                )
                is_valid = False
            if not is_valid:
                citation_failures += 1
            valid_counterfactual[claim.claim_id] = is_valid

        for baseline_claim in trial.baseline_claims:
            claim_id = baseline_claim.claim_id
            baseline_citations = set(baseline_claim.evidence_ids)
            counterfactual_claim = counterfactual_by_id.get(claim_id)
            is_affected = claim_id in affected_ids
            if is_affected:
                affected += 1
                alternatives = baseline_citations - {trial.ablated_evidence_id}
                if not alternatives:
                    unique += 1
                    if counterfactual_claim is None:
                        unique_withdrawn += 1
                        adjusted += 1
                    else:
                        finding(
                            "uniquely_supported_claim_retained",
                            trial.trial_id,
                            claim_id,
                        )
                elif counterfactual_claim is None:
                    adjusted += 1
                else:
                    counterfactual_citations = set(counterfactual_claim.evidence_ids)
                    if (
                        valid_counterfactual[claim_id]
                        and counterfactual_citations
                        and counterfactual_citations <= alternatives
                    ):
                        adjusted += 1
                    else:
                        finding(
                            "affected_claim_not_rebound",
                            trial.trial_id,
                            claim_id,
                        )
            else:
                unaffected += 1
                if counterfactual_claim is None:
                    finding("unaffected_claim_missing", trial.trial_id, claim_id)
                elif (
                    valid_counterfactual[claim_id]
                    and set(counterfactual_claim.evidence_ids) == baseline_citations
                ):
                    unaffected_retained += 1
                else:
                    finding("unaffected_claim_changed", trial.trial_id, claim_id)

    unique_rate = unique_withdrawn / unique if unique else None
    adjustment_rate = adjusted / affected if affected else None
    retention_rate = unaffected_retained / unaffected if unaffected else None

    violations: list[SensitivityViolation] = []
    if len(rows) < active_policy.min_trials:
        violations.append(SensitivityViolation("insufficient_trials"))
    if affected < active_policy.min_affected_claims:
        violations.append(SensitivityViolation("insufficient_affected_claims"))
    if unique < active_policy.min_unique_support_claims:
        violations.append(SensitivityViolation("insufficient_unique_support_claims"))
    if unaffected < active_policy.min_unaffected_claims:
        violations.append(SensitivityViolation("insufficient_unaffected_claims"))
    if irrelevant_trials > active_policy.max_irrelevant_ablation_trials:
        violations.append(SensitivityViolation("irrelevant_ablation_trials_exceeded"))
    if unexpected > active_policy.max_unexpected_claims:
        violations.append(SensitivityViolation("unexpected_claims_exceeded"))
    if citation_failures:
        violations.append(SensitivityViolation("invalid_counterfactual_citations"))
    if unique_rate is not None and unique_rate < active_policy.min_unique_withdrawal_rate:
        violations.append(SensitivityViolation("unique_withdrawal_rate_below_policy"))
    if adjustment_rate is not None and adjustment_rate < active_policy.min_affected_adjustment_rate:
        violations.append(SensitivityViolation("affected_adjustment_rate_below_policy"))
    if retention_rate is not None and retention_rate < active_policy.min_unaffected_retention_rate:
        violations.append(SensitivityViolation("unaffected_retention_rate_below_policy"))

    findings = tuple(
        sorted(
            all_findings,
            key=lambda item: (item.code, item.trial_ref, item.claim_ref or ""),
        )[:MAX_REPORTED_FINDINGS]
    )
    return EvidenceSensitivityReport(
        passed=not violations,
        trials=len(rows),
        affected_claims=affected,
        unique_support_claims=unique,
        unaffected_claims=unaffected,
        irrelevant_ablation_trials=irrelevant_trials,
        unexpected_claims=unexpected,
        counterfactual_citation_failures=citation_failures,
        unique_withdrawal_rate=unique_rate,
        affected_adjustment_rate=adjustment_rate,
        unaffected_retention_rate=retention_rate,
        findings=findings,
        findings_truncated=len(all_findings) > len(findings),
        violations=tuple(violations),
        evidence_digest=_digest(rows, active_policy),
        policy=active_policy,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceSensitivityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _claims(value: Any, name: str) -> tuple[ClaimTrace, ...]:
    if not isinstance(value, list):
        raise EvidenceSensitivityError(f"{name} must be an array")
    claims: list[ClaimTrace] = []
    for claim in value:
        if not isinstance(claim, dict) or set(claim) != {"claim_id", "evidence_ids"}:
            raise EvidenceSensitivityError(
                f"each {name} item must contain exactly claim_id and evidence_ids"
            )
        evidence_ids = claim["evidence_ids"]
        if not isinstance(evidence_ids, list):
            raise EvidenceSensitivityError("evidence_ids must be an array")
        claims.append(
            ClaimTrace(
                claim_id=claim["claim_id"],
                evidence_ids=tuple(evidence_ids),
            )
        )
    return tuple(claims)


def _load(path: Path) -> list[EvidenceAblationTrial]:
    if path.is_symlink():
        raise EvidenceSensitivityError("input must not be a symlink")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise EvidenceSensitivityError(f"input exceeds {MAX_INPUT_BYTES} bytes")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            EvidenceSensitivityError(f"non-finite JSON value: {value}")
        ),
    )
    if not isinstance(payload, list):
        raise EvidenceSensitivityError("input must be a JSON array")
    expected = {
        "ablated_evidence_id",
        "baseline_claims",
        "counterfactual_claims",
        "retained_evidence_ids",
        "trial_id",
    }
    trials: list[EvidenceAblationTrial] = []
    for value in payload:
        if not isinstance(value, dict) or set(value) != expected:
            raise EvidenceSensitivityError(
                "each trial must contain exactly trial_id, ablated_evidence_id, "
                "retained_evidence_ids, baseline_claims, and counterfactual_claims"
            )
        retained = value["retained_evidence_ids"]
        if not isinstance(retained, list):
            raise EvidenceSensitivityError("retained_evidence_ids must be an array")
        trials.append(
            EvidenceAblationTrial(
                trial_id=value["trial_id"],
                ablated_evidence_id=value["ablated_evidence_id"],
                retained_evidence_ids=tuple(retained),
                baseline_claims=_claims(value["baseline_claims"], "baseline_claims"),
                counterfactual_claims=_claims(
                    value["counterfactual_claims"], "counterfactual_claims"
                ),
            )
        )
    return trials


def _write_atomic(path: Path, content: str) -> None:
    if path.is_symlink():
        raise EvidenceSensitivityError("output must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit whether LLM claims respond to counterfactual evidence ablation."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-trials", type=int, default=20)
    parser.add_argument("--min-affected-claims", type=int, default=20)
    parser.add_argument("--min-unique-support-claims", type=int, default=10)
    parser.add_argument("--min-unaffected-claims", type=int, default=20)
    parser.add_argument("--min-unique-withdrawal-rate", type=float, default=0.95)
    parser.add_argument("--min-affected-adjustment-rate", type=float, default=1.0)
    parser.add_argument("--min-unaffected-retention-rate", type=float, default=0.90)
    parser.add_argument("--max-irrelevant-ablation-trials", type=int, default=0)
    parser.add_argument("--max-unexpected-claims", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        policy = EvidenceSensitivityPolicy(
            min_trials=arguments.min_trials,
            min_affected_claims=arguments.min_affected_claims,
            min_unique_support_claims=arguments.min_unique_support_claims,
            min_unaffected_claims=arguments.min_unaffected_claims,
            min_unique_withdrawal_rate=arguments.min_unique_withdrawal_rate,
            min_affected_adjustment_rate=arguments.min_affected_adjustment_rate,
            min_unaffected_retention_rate=arguments.min_unaffected_retention_rate,
            max_irrelevant_ablation_trials=arguments.max_irrelevant_ablation_trials,
            max_unexpected_claims=arguments.max_unexpected_claims,
        )
        report = audit_evidence_sensitivity(_load(arguments.input), policy=policy)
        rendered = json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n"
        if arguments.output:
            _write_atomic(arguments.output, rendered)
        else:
            print(rendered, end="")
    except (
        EvidenceSensitivityError,
        json.JSONDecodeError,
        OSError,
        TypeError,
    ) as exc:
        print(f"evidence sensitivity audit failed: {exc}", file=sys.stderr)
        return INVALID_EVIDENCE_EXIT_CODE
    return PASS_EXIT_CODE if report.passed else POLICY_FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
