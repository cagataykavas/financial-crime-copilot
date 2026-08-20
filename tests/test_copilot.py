from copilot import Action, Copilot, FinancialCrimeCase, Severity, synthetic_case


def test_synthetic_case_escalates():
    case = synthetic_case()
    recommendation = Copilot().recommend(case)
    assert recommendation.recommended_action is Action.ESCALATE
    assert recommendation.priority >= 80
    assert recommendation.evidence_ids


def test_no_signal_case_requires_human_verified_closure():
    case = FinancialCrimeCase(
        case_id="empty",
        subject_id="demo",
        subject_type="retail",
        opened_at="2026-01-01T00:00:00+00:00",
        customer_impact=Severity.LOW,
    )
    recommendation = Copilot().recommend(case)
    assert recommendation.recommended_action is Action.CLOSE
    assert not recommendation.allowed_to_auto_execute


def test_reviewer_override_is_audited():
    case = synthetic_case()
    copilot = Copilot()
    recommendation = copilot.recommend(case)
    copilot.reviewer_decide(
        case,
        recommendation,
        action=Action.REQUEST_INFORMATION,
        reason="Need synthetic source-of-funds context.",
        reviewer_id="reviewer-test",
    )
    event = case.audit_events[-1]
    assert event["overridden"] is True
    assert event["agent_recommendation"] == "escalate"
    assert event["reviewer_action"] == "request_information"


def test_material_escalation_not_auto_executed():
    recommendation = Copilot().recommend(synthetic_case())
    assert recommendation.recommended_action is Action.ESCALATE
    assert recommendation.allowed_to_auto_execute is False
