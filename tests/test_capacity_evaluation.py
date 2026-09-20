import json
from math import nan

import pytest

from capacity_evaluation import AlertOutcome, CapacityPolicy, evaluate_sufficient_review_capacity


def rows(scores_and_labels):
    return [
        AlertOutcome(f"a{index:02d}", score, label)
        for index, (score, label) in enumerate(scores_and_labels)
    ]


def policy(**overrides):
    values = {
        "review_fraction": 0.2,
        "min_alerts": 10,
        "min_positives": 2,
        "min_precision_at_capacity": 0.5,
        "min_recall_at_capacity": 0.5,
        "min_lift_over_random": 2.0,
    }
    values.update(overrides)
    return CapacityPolicy(**values)


def test_good_ranking_passes_and_reports_operational_metrics():
    data = rows([(0.99, True), (0.90, True)] + [(0.8 - i / 10, False) for i in range(8)])
    report = evaluate_sufficient_review_capacity(data, policy())
    assert report.passed
    assert (report.review_count, report.reviewed_positives, report.missed_positives) == (2, 2, 0)
    assert report.precision_at_capacity == report.recall_at_capacity == 1.0
    assert report.lift_over_random == 5.0
    assert report.workload_reduction == 0.8


def test_bad_ranking_returns_all_release_reasons():
    data = rows([(0.99, False), (0.90, False)] + [(0.8 - i / 10, i >= 6) for i in range(8)])
    report = evaluate_sufficient_review_capacity(data, policy())
    assert not report.passed
    assert report.reasons == ("precision_below_floor", "recall_below_floor", "lift_below_floor")
    assert report.missed_positives == 2


def test_capacity_uses_ceiling_and_deterministic_tie_breaking():
    data = rows([(0.5, True), (0.5, False), (0.5, False)] + [(0.1, False)] * 7)
    report = evaluate_sufficient_review_capacity(
        data, policy(review_fraction=0.21, min_positives=1, min_lift_over_random=1)
    )
    assert report.review_count == 3
    assert report.reviewed_alert_ids == ("a00", "a01", "a02")


def test_report_is_json_serializable():
    data = rows([(1.0, True), (0.9, True)] + [(0.1, False)] * 8)
    payload = json.loads(json.dumps(evaluate_sufficient_review_capacity(data, policy()).to_dict()))
    assert payload["passed"] is True
    assert payload["reviewed_alert_ids"] == ["a00", "a01"]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (rows([(0.5, True)] * 9), "at least 10 alerts"),
        (rows([(0.5, True)] + [(0.4, False)] * 9), "at least 2 positives"),
    ],
)
def test_sparse_evidence_fails_closed(data, message):
    with pytest.raises(ValueError, match=message):
        evaluate_sufficient_review_capacity(data, policy())


def test_duplicate_ids_are_rejected():
    data = rows([(1.0, True), (0.9, True)] + [(0.1, False)] * 8)
    data[1] = AlertOutcome("a00", 0.9, True)
    with pytest.raises(ValueError, match="duplicate alert_id"):
        evaluate_sufficient_review_capacity(data, policy())


def test_non_finite_scores_are_rejected():
    data = rows([(1.0, True), (0.9, True)] + [(0.1, False)] * 8)
    data[3] = AlertOutcome("a03", nan, False)
    with pytest.raises(ValueError, match="risk_score must be finite"):
        evaluate_sufficient_review_capacity(data, policy())


@pytest.mark.parametrize(
    "bad_policy",
    [
        policy(review_fraction=0),
        policy(review_fraction=1.1),
        policy(min_alerts=0),
        policy(min_positives=0),
        policy(min_lift_over_random=0),
    ],
)
def test_invalid_policy_is_rejected(bad_policy):
    with pytest.raises(ValueError):
        evaluate_sufficient_review_capacity([], bad_policy)
