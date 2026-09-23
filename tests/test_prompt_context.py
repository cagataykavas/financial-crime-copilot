from __future__ import annotations

from dataclasses import replace

import pytest

from copilot import Copilot, EvidenceKind, synthetic_case
from service.prompt_context import (
    ContextProjectionError,
    PromptContextPolicy,
    build_prompt_context,
)

PSEUDONYM_KEY = b"test-only-context-key-material-32-bytes-minimum"


def _case_and_recommendation():
    case = synthetic_case()
    recommendation = Copilot().recommend(case)
    return case, recommendation


def test_projection_removes_raw_identifiers_free_text_and_unapproved_attributes():
    case, recommendation = _case_and_recommendation()
    original = case.evidence[0]
    case.evidence[0] = replace(
        original,
        attributes={
            **original.attributes,
            "account_number": "TR00-SHOULD-NOT-LEAVE",
            "email": "synthetic@example.test",
        },
    )

    bundle = build_prompt_context(
        case,
        recommendation,
        pseudonym_key=PSEUDONYM_KEY,
        key_id="test-key-v1",
    )
    serialized = bundle.canonical_model_json
    context = bundle.model_context

    assert context["trust"] == "untrusted_case_evidence"
    assert context["evidence"][0]["attributes"]["total_amount"] == 48250
    assert case.case_id not in serialized
    assert case.subject_id not in serialized
    assert "tx-001" not in serialized
    assert "sig-velocity" not in serialized
    assert "TR00-SHOULD-NOT-LEAVE" not in serialized
    assert "synthetic@example.test" not in serialized
    assert original.summary not in serialized
    assert original.source not in serialized


def test_local_manifest_preserves_alias_provenance_and_exact_digest():
    case, recommendation = _case_and_recommendation()

    bundle = build_prompt_context(
        case,
        recommendation,
        pseudonym_key=PSEUDONYM_KEY,
        key_id="test-key-v1",
    )

    assert bundle.manifest.case_id == case.case_id
    assert bundle.manifest.policy_id == "minimum-necessary-v1"
    assert bundle.manifest.evidence_refs[0] == ("e1", recommendation.evidence_ids[0])
    assert bundle.manifest.signal_refs[0] == ("s1", recommendation.key_signal_ids[0])
    assert bundle.manifest.context_bytes == len(bundle.canonical_model_json.encode("utf-8"))
    assert len(bundle.manifest.context_sha256) == 64


def test_pseudonyms_are_stable_and_purpose_scoped():
    case, recommendation = _case_and_recommendation()
    first = build_prompt_context(
        case,
        recommendation,
        pseudonym_key=PSEUDONYM_KEY,
        key_id="test-key-v1",
    ).model_context
    repeated = build_prompt_context(
        case,
        recommendation,
        pseudonym_key=PSEUDONYM_KEY,
        key_id="test-key-v1",
    ).model_context
    other_purpose = build_prompt_context(
        case,
        recommendation,
        pseudonym_key=PSEUDONYM_KEY,
        key_id="test-key-v1",
        policy=PromptContextPolicy(purpose="quality_assurance_review"),
    ).model_context

    assert first["subject_ref"] == repeated["subject_ref"]
    assert first["case_ref"] == repeated["case_ref"]
    assert first["subject_ref"] != other_purpose["subject_ref"]
    assert first["case_ref"] != other_purpose["case_ref"]


def test_context_is_deterministic_across_attribute_insertion_order():
    case, recommendation = _case_and_recommendation()
    first = build_prompt_context(
        case,
        recommendation,
        pseudonym_key=PSEUDONYM_KEY,
        key_id="test-key-v1",
    )
    evidence = case.evidence[0]
    case.evidence[0] = replace(evidence, attributes=dict(reversed(evidence.attributes.items())))
    second = build_prompt_context(
        case,
        recommendation,
        pseudonym_key=PSEUDONYM_KEY,
        key_id="test-key-v1",
    )

    assert first.canonical_model_json == second.canonical_model_json
    assert first.manifest.context_sha256 == second.manifest.context_sha256


def test_model_context_access_cannot_mutate_canonical_bundle():
    case, recommendation = _case_and_recommendation()
    bundle = build_prompt_context(
        case,
        recommendation,
        pseudonym_key=PSEUDONYM_KEY,
        key_id="test-key-v1",
    )

    first = bundle.model_context
    first["evidence"][0]["attributes"]["count"] = 999

    assert bundle.model_context["evidence"][0]["attributes"]["count"] == 5


@pytest.mark.parametrize(
    ("field_name", "value", "code"),
    [
        ("case_id", "different-case", "case_mismatch"),
        ("evidence_ids", ("missing",), "missing_selected_evidence"),
        ("key_signal_ids", ("missing",), "missing_selected_signal"),
        ("evidence_ids", ("tx-001", "tx-001"), "duplicate_selected_evidence"),
        ("key_signal_ids", ("sig-velocity", "sig-velocity"), "duplicate_selected_signal"),
    ],
)
def test_broken_recommendation_contracts_fail_closed(field_name, value, code):
    case, recommendation = _case_and_recommendation()
    recommendation = replace(recommendation, **{field_name: value})

    with pytest.raises(ContextProjectionError) as caught:
        build_prompt_context(
            case,
            recommendation,
            pseudonym_key=PSEUDONYM_KEY,
            key_id="test-key-v1",
        )

    assert caught.value.code == code


def test_duplicate_case_evidence_fails_before_aliasing():
    case, recommendation = _case_and_recommendation()
    case.evidence.append(case.evidence[0])

    with pytest.raises(ContextProjectionError) as caught:
        build_prompt_context(
            case,
            recommendation,
            pseudonym_key=PSEUDONYM_KEY,
            key_id="test-key-v1",
        )

    assert caught.value.code == "duplicate_case_evidence"


def test_signal_cannot_cite_evidence_omitted_from_model_context():
    case, recommendation = _case_and_recommendation()
    reduced = replace(recommendation, evidence_ids=(recommendation.evidence_ids[0],))

    with pytest.raises(ContextProjectionError) as caught:
        build_prompt_context(
            case,
            reduced,
            pseudonym_key=PSEUDONYM_KEY,
            key_id="test-key-v1",
        )

    assert caught.value.code == "signal_evidence_not_selected"


@pytest.mark.parametrize(
    ("attribute_value", "code"),
    [
        ([1, 2], "non_scalar_attribute"),
        (float("inf"), "non_finite_attribute"),
    ],
)
def test_unsafe_allowlisted_attribute_values_fail_closed(attribute_value, code):
    case, recommendation = _case_and_recommendation()
    evidence = case.evidence[0]
    case.evidence[0] = replace(
        evidence,
        attributes={**evidence.attributes, "count": attribute_value},
    )

    with pytest.raises(ContextProjectionError) as caught:
        build_prompt_context(
            case,
            recommendation,
            pseudonym_key=PSEUDONYM_KEY,
            key_id="test-key-v1",
        )

    assert caught.value.code == code


def test_invalid_confidence_is_rejected_at_projection_boundary():
    case, recommendation = _case_and_recommendation()
    case.evidence[0] = replace(case.evidence[0], confidence=float("nan"))

    with pytest.raises(ContextProjectionError) as caught:
        build_prompt_context(
            case,
            recommendation,
            pseudonym_key=PSEUDONYM_KEY,
            key_id="test-key-v1",
        )

    assert caught.value.code == "invalid_evidence_confidence"


@pytest.mark.parametrize("event_time", ["not-a-time", "2026-01-01T00:00:00"])
def test_event_time_must_be_parseable_and_timezone_aware(event_time):
    case, recommendation = _case_and_recommendation()
    case.evidence[0] = replace(case.evidence[0], event_time=event_time)

    with pytest.raises(ContextProjectionError) as caught:
        build_prompt_context(
            case,
            recommendation,
            pseudonym_key=PSEUDONYM_KEY,
            key_id="test-key-v1",
        )

    assert caught.value.code == "invalid_event_time"


def test_event_time_is_normalized_to_utc():
    case, recommendation = _case_and_recommendation()
    case.evidence[0] = replace(case.evidence[0], event_time="2026-01-01T03:00:00+03:00")

    context = build_prompt_context(
        case,
        recommendation,
        pseudonym_key=PSEUDONYM_KEY,
        key_id="test-key-v1",
    ).model_context

    assert context["evidence"][0]["event_time"] == "2026-01-01T00:00:00+00:00"


def test_invalid_recommendation_priority_is_rejected():
    case, recommendation = _case_and_recommendation()

    with pytest.raises(ContextProjectionError) as caught:
        build_prompt_context(
            case,
            replace(recommendation, priority=101),
            pseudonym_key=PSEUDONYM_KEY,
            key_id="test-key-v1",
        )

    assert caught.value.code == "invalid_priority"


@pytest.mark.parametrize(
    ("policy", "code"),
    [
        (PromptContextPolicy(max_evidence=2), "evidence_budget_exceeded"),
        (PromptContextPolicy(max_signals=2), "signal_budget_exceeded"),
        (PromptContextPolicy(max_context_bytes=100), "context_size_exceeded"),
    ],
)
def test_projection_budgets_fail_closed(policy, code):
    case, recommendation = _case_and_recommendation()

    with pytest.raises(ContextProjectionError) as caught:
        build_prompt_context(
            case,
            recommendation,
            pseudonym_key=PSEUDONYM_KEY,
            key_id="test-key-v1",
            policy=policy,
        )

    assert caught.value.code == code


def test_string_budget_applies_to_selected_structured_values():
    case, recommendation = _case_and_recommendation()
    case.evidence[0] = replace(
        case.evidence[0],
        attributes={**case.evidence[0].attributes, "currency": "TOO-LONG"},
    )

    with pytest.raises(ContextProjectionError) as caught:
        build_prompt_context(
            case,
            recommendation,
            pseudonym_key=PSEUDONYM_KEY,
            key_id="test-key-v1",
            policy=PromptContextPolicy(max_string_bytes=4),
        )

    assert caught.value.code == "string_too_large"


@pytest.mark.parametrize(
    ("key", "key_id"),
    [
        (b"too-short", "valid-key"),
        (PSEUDONYM_KEY, "invalid key id"),
    ],
)
def test_pseudonym_key_contract_is_fail_closed(key, key_id):
    case, recommendation = _case_and_recommendation()

    with pytest.raises(ValueError):
        build_prompt_context(
            case,
            recommendation,
            pseudonym_key=key,
            key_id=key_id,
        )


def test_custom_policy_can_remove_event_time_and_all_attributes():
    case, recommendation = _case_and_recommendation()
    policy = PromptContextPolicy(
        include_event_time=False,
        allowed_attributes={kind: frozenset() for kind in EvidenceKind},
    )

    bundle = build_prompt_context(
        case,
        recommendation,
        pseudonym_key=PSEUDONYM_KEY,
        key_id="test-key-v1",
        policy=policy,
    )

    assert all("event_time" not in item for item in bundle.model_context["evidence"])
    assert all(not item["attributes"] for item in bundle.model_context["evidence"])
    assert all(count >= 0 for _, count in bundle.manifest.excluded_attributes)
