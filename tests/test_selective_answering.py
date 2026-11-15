import json
from pathlib import Path

import pytest

from service.selective_answering import (
    SelectiveAnswerObservation,
    SelectiveAnswerPolicy,
    evaluate_selective_answering,
    main,
)


def observation(
    index: int,
    *,
    confidence: float,
    answerable: bool,
    supported: bool,
) -> SelectiveAnswerObservation:
    return SelectiveAnswerObservation(
        case_id=f"case-{index:04d}",
        confidence=confidence,
        answerable=answerable,
        answer_supported=supported,
    )


def passing_observations() -> list[SelectiveAnswerObservation]:
    rows = [
        observation(index, confidence=0.95, answerable=True, supported=True) for index in range(285)
    ]
    rows.extend(
        observation(index, confidence=0.40, answerable=True, supported=True)
        for index in range(285, 300)
    )
    rows.extend(
        observation(index, confidence=0.20, answerable=False, supported=False)
        for index in range(300, 400)
    )
    return rows


def permissive_policy(**overrides: object) -> SelectiveAnswerPolicy:
    values = {
        "minimum_observations": 4,
        "minimum_answerable": 2,
        "minimum_unanswerable": 2,
        "maximum_selective_risk": 1.0,
        "maximum_false_answer_rate": 1.0,
        "minimum_answerable_coverage": 0.0,
    }
    values.update(overrides)
    return SelectiveAnswerPolicy(**values)


def test_reviewed_population_passes_confidence_bounded_policy() -> None:
    report = evaluate_selective_answering(passing_observations())

    assert report.decision == "pass"
    assert report.reasons == ()
    assert report.answered == 285
    assert report.selective_risk.successes == 0
    assert report.false_answer_rate.successes == 0
    assert report.answerable_coverage.lower_bound >= 0.80


def test_unsupported_high_confidence_answers_fail_selective_risk() -> None:
    rows = passing_observations()
    rows[:30] = [
        observation(index, confidence=0.95, answerable=True, supported=False) for index in range(30)
    ]

    report = evaluate_selective_answering(rows)

    assert report.decision == "fail"
    assert "selective_risk_upper_bound_exceeded" in report.reasons


def test_high_confidence_unanswerable_cases_fail_false_answer_rate() -> None:
    rows = passing_observations()
    rows[-10:] = [
        observation(index, confidence=0.95, answerable=False, supported=False)
        for index in range(390, 400)
    ]

    report = evaluate_selective_answering(rows)

    assert "false_answer_rate_upper_bound_exceeded" in report.reasons


def test_over_abstention_fails_answerable_coverage() -> None:
    rows = passing_observations()
    rows[:100] = [
        observation(index, confidence=0.20, answerable=True, supported=True) for index in range(100)
    ]

    report = evaluate_selective_answering(rows)

    assert "answerable_coverage_lower_bound_below_minimum" in report.reasons


def test_confidence_bounds_prevent_tiny_sample_false_assurance() -> None:
    rows = [
        observation(0, confidence=0.9, answerable=True, supported=True),
        observation(1, confidence=0.9, answerable=True, supported=True),
        observation(2, confidence=0.1, answerable=False, supported=False),
        observation(3, confidence=0.1, answerable=False, supported=False),
    ]
    policy = permissive_policy(
        maximum_selective_risk=0.20,
        maximum_false_answer_rate=0.20,
        minimum_answerable_coverage=0.80,
    )

    report = evaluate_selective_answering(rows, policy)

    assert report.selective_risk.rate == 0.0
    assert report.selective_risk.upper_bound > 0.20
    assert report.decision == "fail"


def test_digest_is_invariant_to_input_order() -> None:
    rows = passing_observations()

    forward = evaluate_selective_answering(rows)
    reverse = evaluate_selective_answering(list(reversed(rows)))

    assert forward.evidence_digest == reverse.evidence_digest


def test_digest_binds_policy() -> None:
    rows = passing_observations()

    first = evaluate_selective_answering(rows)
    second = evaluate_selective_answering(rows, SelectiveAnswerPolicy(confidence_threshold=0.76))

    assert first.evidence_digest != second.evidence_digest


def test_duplicate_case_ids_fail_closed() -> None:
    rows = passing_observations()
    rows[-1] = observation(0, confidence=0.2, answerable=False, supported=False)

    with pytest.raises(ValueError, match="duplicate case_id"):
        evaluate_selective_answering(rows)


@pytest.mark.parametrize("confidence", [float("nan"), -0.1, 1.1, True])
def test_invalid_confidence_fails_closed(confidence: float) -> None:
    rows = passing_observations()
    rows[0] = observation(0, confidence=confidence, answerable=True, supported=True)

    with pytest.raises(ValueError, match="confidence"):
        evaluate_selective_answering(rows)


def test_unanswerable_supported_label_fails_closed() -> None:
    rows = passing_observations()
    rows[-1] = observation(399, confidence=0.2, answerable=False, supported=True)

    with pytest.raises(ValueError, match="unanswerable"):
        evaluate_selective_answering(rows)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_level": 1.0},
        {"minimum_observations": True},
        {"minimum_unanswerable": 0},
        {"maximum_selective_risk": float("nan")},
    ],
)
def test_invalid_policy_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SelectiveAnswerPolicy(**kwargs)


def write_observations(path: Path, rows: list[SelectiveAnswerObservation]) -> None:
    path.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "case_id": item.case_id,
                        "confidence": item.confidence,
                        "answerable": item.answerable,
                        "answer_supported": item.answer_supported,
                    }
                    for item in rows
                ]
            }
        ),
        encoding="utf-8",
    )


def test_cli_writes_passing_report(tmp_path: Path) -> None:
    input_path = tmp_path / "observations.json"
    output_path = tmp_path / "report.json"
    write_observations(input_path, passing_observations())

    exit_code = main([str(input_path), "--output", str(output_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["decision"] == "pass"
    assert len(payload["evidence_digest"]) == 64
    assert payload["policy"]["confidence_threshold"] == 0.75


def test_cli_returns_distinct_policy_failure(tmp_path: Path) -> None:
    input_path = tmp_path / "observations.json"
    output_path = tmp_path / "report.json"
    rows = passing_observations()
    rows[-10:] = [
        observation(index, confidence=0.95, answerable=False, supported=False)
        for index in range(390, 400)
    ]
    write_observations(input_path, rows)

    exit_code = main([str(input_path), "--output", str(output_path)])

    assert exit_code == 2
    assert json.loads(output_path.read_text(encoding="utf-8"))["decision"] == "fail"


def test_cli_rejects_duplicate_json_fields_without_output(tmp_path: Path) -> None:
    input_path = tmp_path / "observations.json"
    output_path = tmp_path / "report.json"
    input_path.write_text('{"observations":[],"observations":[]}', encoding="utf-8")

    exit_code = main([str(input_path), "--output", str(output_path)])

    assert exit_code == 3
    assert not output_path.exists()
