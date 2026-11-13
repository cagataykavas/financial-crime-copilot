from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from service.evidence_sensitivity import (
    INVALID_EVIDENCE_EXIT_CODE,
    PASS_EXIT_CODE,
    POLICY_FAILURE_EXIT_CODE,
    ClaimTrace,
    EvidenceAblationTrial,
    EvidenceSensitivityError,
    EvidenceSensitivityPolicy,
    audit_evidence_sensitivity,
    main,
)


def _claim(claim_id: str, *evidence_ids: str) -> ClaimTrace:
    return ClaimTrace(claim_id, evidence_ids)


def _trials(count: int = 20) -> list[EvidenceAblationTrial]:
    return [
        EvidenceAblationTrial(
            trial_id=f"trial-{index}",
            ablated_evidence_id=f"removed-{index}",
            retained_evidence_ids=(f"alternate-{index}", f"stable-{index}"),
            baseline_claims=(
                _claim(f"unique-{index}", f"removed-{index}"),
                _claim(
                    f"shared-{index}",
                    f"removed-{index}",
                    f"alternate-{index}",
                ),
                _claim(f"stable-claim-{index}", f"stable-{index}"),
            ),
            counterfactual_claims=(
                _claim(f"shared-{index}", f"alternate-{index}"),
                _claim(f"stable-claim-{index}", f"stable-{index}"),
            ),
        )
        for index in range(count)
    ]


def _codes(report) -> set[str]:
    return {violation.code for violation in report.violations}


def test_valid_ablation_suite_withdraws_unique_claims_and_preserves_controls() -> None:
    report = audit_evidence_sensitivity(_trials())

    assert report.passed
    assert report.trials == 20
    assert report.affected_claims == 40
    assert report.unique_support_claims == 20
    assert report.unaffected_claims == 20
    assert report.unique_withdrawal_rate == 1.0
    assert report.affected_adjustment_rate == 1.0
    assert report.unaffected_retention_rate == 1.0
    assert report.findings == ()
    rendered = json.dumps(report.as_dict())
    assert "trial-0" not in rendered
    assert "removed-0" not in rendered


def test_retaining_uniquely_supported_claim_fails_sensitivity_gates() -> None:
    trials = _trials()
    first = trials[0]
    trials[0] = replace(
        first,
        counterfactual_claims=first.counterfactual_claims + (_claim("unique-0", "alternate-0"),),
    )
    policy = EvidenceSensitivityPolicy(
        min_unique_withdrawal_rate=1.0,
        min_affected_adjustment_rate=1.0,
    )

    report = audit_evidence_sensitivity(trials, policy=policy)

    assert not report.passed
    assert _codes(report) >= {
        "unique_withdrawal_rate_below_policy",
        "affected_adjustment_rate_below_policy",
    }
    assert "uniquely_supported_claim_retained" in {finding.code for finding in report.findings}


def test_unaffected_claim_loss_is_detected_as_generic_output_collapse() -> None:
    trials = _trials()
    for index in range(3):
        trial = trials[index]
        trials[index] = replace(
            trial,
            counterfactual_claims=tuple(
                claim
                for claim in trial.counterfactual_claims
                if claim.claim_id != f"stable-claim-{index}"
            ),
        )
    policy = EvidenceSensitivityPolicy(min_unaffected_retention_rate=0.90)

    report = audit_evidence_sensitivity(trials, policy=policy)

    assert "unaffected_retention_rate_below_policy" in _codes(report)
    assert report.unaffected_retention_rate == pytest.approx(0.85)


def test_new_claims_and_forbidden_citations_fail_closed() -> None:
    trials = _trials()
    first = trials[0]
    trials[0] = replace(
        first,
        counterfactual_claims=first.counterfactual_claims
        + (
            _claim("new-claim", "alternate-0"),
            _claim("unique-0", "removed-0"),
        ),
    )

    report = audit_evidence_sensitivity(trials)

    finding_codes = {finding.code for finding in report.findings}
    assert "unexpected_claims_exceeded" in _codes(report)
    assert "invalid_counterfactual_citations" in _codes(report)
    assert report.counterfactual_citation_failures == 1
    assert "unexpected_claim" in finding_codes
    assert "counterfactual_cites_ablated_evidence" in finding_codes


def test_irrelevant_ablation_and_duplicate_trials_are_distinguished() -> None:
    trials = _trials()
    first = trials[0]
    trials[0] = replace(
        first,
        ablated_evidence_id="unused",
        retained_evidence_ids=first.retained_evidence_ids + ("removed-0",),
    )

    report = audit_evidence_sensitivity(trials)
    assert "irrelevant_ablation_trials_exceeded" in _codes(report)

    with pytest.raises(EvidenceSensitivityError, match="trial_id values must be unique"):
        audit_evidence_sensitivity([trials[0], trials[0]])


def test_input_order_does_not_change_report_or_digest() -> None:
    trials = _trials()
    reordered = [
        replace(
            trial,
            retained_evidence_ids=tuple(reversed(trial.retained_evidence_ids)),
            baseline_claims=tuple(reversed(trial.baseline_claims)),
            counterfactual_claims=tuple(reversed(trial.counterfactual_claims)),
        )
        for trial in reversed(trials)
    ]

    assert audit_evidence_sensitivity(trials) == audit_evidence_sensitivity(reordered)


def test_cli_distinguishes_pass_policy_failure_and_invalid_input(tmp_path: Path) -> None:
    trials = _trials()
    input_path = tmp_path / "trials.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(json.dumps([asdict(trial) for trial in trials]), encoding="utf-8")

    assert main([str(input_path), "--output", str(output_path)]) == PASS_EXIT_CODE
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is True
    assert main([str(input_path), "--min-unaffected-claims", "21"]) == POLICY_FAILURE_EXIT_CODE

    input_path.write_text(
        '[{"trial_id":"a","trial_id":"b"}]',
        encoding="utf-8",
    )
    assert main([str(input_path)]) == INVALID_EVIDENCE_EXIT_CODE
