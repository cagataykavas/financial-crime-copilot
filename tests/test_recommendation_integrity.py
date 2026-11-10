from __future__ import annotations

from dataclasses import replace

import pytest

from copilot import (
    Action,
    Copilot,
    CopilotRecommendation,
    FinancialCrimeCase,
    synthetic_case,
)
from governance import (
    MAX_NARRATIVE_BYTES,
    PolicyGate,
    PolicyOutcome,
    validate_recommendation_integrity,
)


def recommendation() -> tuple[FinancialCrimeCase, CopilotRecommendation]:
    case = synthetic_case()
    return case, Copilot().recommend(case)


def issue_codes(case: FinancialCrimeCase, candidate: CopilotRecommendation) -> set[str]:
    return {issue.code for issue in validate_recommendation_integrity(case, candidate)}


def test_generated_recommendation_has_exact_evidence_closure() -> None:
    case, candidate = recommendation()

    assert validate_recommendation_integrity(case, candidate) == ()


def test_no_signal_recommendation_cites_the_available_evidence() -> None:
    case = synthetic_case()
    case.signals.clear()
    candidate = Copilot().recommend(case)

    assert validate_recommendation_integrity(case, candidate) == ()


@pytest.mark.parametrize(
    ("changes", "expected_code"),
    [
        ({"case_id": "other-case"}, "recommendation_case_mismatch"),
        ({"key_signal_ids": ("unknown-signal",)}, "unknown_recommendation_signal"),
        ({"evidence_ids": ("unknown-evidence",)}, "unknown_recommendation_evidence"),
        ({"key_signal_ids": ()}, "missing_recommendation_signals"),
        ({"priority": True}, "invalid_recommendation_priority"),
        ({"recommended_action": "escalate"}, "invalid_recommended_action"),
        ({"allowed_to_auto_execute": "yes"}, "invalid_automatic_execution_flag"),
        (
            {"uncertainty_notes": ["not-a-tuple"]},
            "invalid_recommendation_uncertainty_notes_container",
        ),
    ],
)
def test_rejects_recommendation_identity_and_shape_drift(
    changes: dict[str, object],
    expected_code: str,
) -> None:
    case, candidate = recommendation()

    mutated = replace(candidate, **changes)

    assert expected_code in issue_codes(case, mutated)


def test_rejects_omitted_extra_and_duplicate_evidence() -> None:
    case, candidate = recommendation()
    case.evidence.append(replace(case.evidence[0], evidence_id="unused-evidence"))
    omitted = replace(candidate, evidence_ids=candidate.evidence_ids[:-1])
    extra = replace(candidate, evidence_ids=(*candidate.evidence_ids, "unused-evidence"))
    duplicated = replace(
        candidate, evidence_ids=(*candidate.evidence_ids, candidate.evidence_ids[0])
    )

    assert "recommendation_evidence_closure_mismatch" in issue_codes(case, omitted)
    assert "recommendation_evidence_closure_mismatch" in issue_codes(case, extra)
    assert "duplicate_recommendation_evidence_ids" in issue_codes(case, duplicated)


def test_rejects_unbounded_text_and_unsafe_auto_execution_claim() -> None:
    case, candidate = recommendation()
    mutated = replace(
        candidate,
        narrative="x" * (MAX_NARRATIVE_BYTES + 1),
        recommended_action=Action.ESCALATE,
        allowed_to_auto_execute=True,
    )

    codes = issue_codes(case, mutated)

    assert "invalid_recommendation_narrative" in codes
    assert "invalid_automatic_execution_claim" in codes


def test_policy_gate_blocks_mutated_recommendation_before_authority_checks() -> None:
    case, candidate = recommendation()
    mutated = replace(candidate, evidence_ids=("invented-evidence",))

    decision = PolicyGate().evaluate(case, mutated)

    assert decision.outcome is PolicyOutcome.BLOCK
    assert decision.reasons == ("case_integrity_failed",)
    assert {issue.code for issue in decision.integrity_issues} >= {
        "unknown_recommendation_evidence",
        "recommendation_evidence_closure_mismatch",
    }
