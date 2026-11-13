from __future__ import annotations

import json
from pathlib import Path

import pytest

from service.context_order_evaluation import (
    ContextOrderTrial,
    EvaluationInputError,
    evaluate_context_order,
    load_trials,
    main,
)


def _trial(
    trial_id: str,
    order: tuple[str, ...],
    *,
    baseline: bool = False,
    action: str = "escalate",
    cited: frozenset[str] = frozenset({"ev-1", "ev-2"}),
    case_id: str = "case-1",
) -> ContextOrderTrial:
    return ContextOrderTrial(case_id, trial_id, baseline, order, action, cited)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_passes_stable_action_and_citations_across_reordering() -> None:
    report = evaluate_context_order(
        [
            _trial("baseline", ("ev-1", "ev-2"), baseline=True),
            _trial("reverse", ("ev-2", "ev-1")),
        ]
    )

    assert report.ok
    assert report.cases[0].action_stability_rate == 1.0
    assert report.cases[0].minimum_citation_jaccard == 1.0


def test_fails_when_action_changes_with_order() -> None:
    report = evaluate_context_order(
        [
            _trial("baseline", ("ev-1", "ev-2"), baseline=True),
            _trial("reverse", ("ev-2", "ev-1"), action="close"),
        ]
    )

    assert not report.ok
    assert report.cases[0].action_stability_rate == 0.0
    assert [issue.code for issue in report.cases[0].issues] == ["trial.action_changed"]


def test_gates_citation_set_instability() -> None:
    report = evaluate_context_order(
        [
            _trial("baseline", ("ev-1", "ev-2"), baseline=True),
            _trial("reverse", ("ev-2", "ev-1"), cited=frozenset({"ev-1"})),
        ],
        minimum_citation_jaccard=0.75,
    )

    assert not report.ok
    assert report.cases[0].minimum_citation_jaccard == 0.5
    assert report.cases[0].issues[0].code == "trial.citation_instability"


def test_rejects_membership_change_instead_of_conflating_it_with_order() -> None:
    report = evaluate_context_order(
        [
            _trial("baseline", ("ev-1", "ev-2"), baseline=True),
            _trial("changed", ("ev-2", "ev-3"), cited=frozenset({"ev-2"})),
        ]
    )

    assert not report.ok
    assert report.cases[0].comparable_variants == 0
    assert report.cases[0].issues[0].code == "trial.evidence_membership_changed"


def test_rejects_duplicate_baseline_as_fake_permutation() -> None:
    report = evaluate_context_order(
        [
            _trial("baseline", ("ev-1", "ev-2"), baseline=True),
            _trial("copy", ("ev-1", "ev-2")),
        ]
    )

    assert not report.ok
    assert report.cases[0].issues[0].code == "case.no_reordering"


def test_requires_exactly_one_baseline() -> None:
    report = evaluate_context_order(
        [
            _trial("one", ("ev-1", "ev-2")),
            _trial("two", ("ev-2", "ev-1")),
        ]
    )

    assert not report.ok
    assert report.cases[0].issues[0].code == "case.baseline_count"


def test_load_trials_rejects_citation_outside_context(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    _write_jsonl(
        path,
        [
            {
                "case_id": "case-1",
                "trial_id": "baseline",
                "is_baseline": True,
                "evidence_order": ["ev-1"],
                "action": "escalate",
                "cited_evidence": ["missing"],
            }
        ],
    )

    with pytest.raises(EvaluationInputError, match="absent evidence"):
        load_trials(path)


def test_load_trials_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "trials.jsonl"
    path.write_text('{"case_id":"one","case_id":"two"}\n', encoding="utf-8")

    with pytest.raises(EvaluationInputError, match="duplicate JSON key"):
        load_trials(path)


def test_cli_writes_atomic_failure_report_and_returns_two(tmp_path: Path) -> None:
    input_path = tmp_path / "trials.jsonl"
    output_path = tmp_path / "report.json"
    rows = [
        {
            "case_id": "case-1",
            "trial_id": "baseline",
            "is_baseline": True,
            "evidence_order": ["ev-1", "ev-2"],
            "action": "escalate",
            "cited_evidence": ["ev-1", "ev-2"],
        },
        {
            "case_id": "case-1",
            "trial_id": "reverse",
            "is_baseline": False,
            "evidence_order": ["ev-2", "ev-1"],
            "action": "close",
            "cited_evidence": ["ev-1", "ev-2"],
        },
    ]
    _write_jsonl(input_path, rows)

    exit_code = main([str(input_path), "--output", str(output_path)])

    assert exit_code == 2
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert not report["ok"]
    assert report["cases"][0]["issues"][0]["code"] == "trial.action_changed"
