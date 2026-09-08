from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from copilot import Action, Copilot, synthetic_case
from service.store import CaseRepository, DecisionConflict


def _review(
    repository: CaseRepository,
    *,
    action: Action,
    reviewer_id: str,
    barrier: threading.Barrier,
) -> tuple[Action, int]:
    snapshot = repository.get_snapshot("FC-DEMO-20418")
    assert snapshot is not None
    recommendation = Copilot().recommend(snapshot.case)
    Copilot.reviewer_decide(
        snapshot.case,
        recommendation,
        action=action,
        reason=f"Concurrent synthetic decision from {reviewer_id}.",
        reviewer_id=reviewer_id,
    )
    barrier.wait(timeout=2)
    version = repository.commit_review(
        snapshot.case,
        known_audit_count=0,
        expected_version=snapshot.version,
    )
    return action, version


def test_concurrent_reviewers_cannot_both_resolve_one_case(tmp_path: Path) -> None:
    repository = CaseRepository(tmp_path / "concurrent.db")
    repository.upsert(synthetic_case())
    barrier = threading.Barrier(2)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                _review,
                repository,
                action=Action.ESCALATE,
                reviewer_id="reviewer-a",
                barrier=barrier,
            ),
            pool.submit(
                _review,
                repository,
                action=Action.RESTRICT,
                reviewer_id="reviewer-b",
                barrier=barrier,
            ),
        ]

    successes: list[tuple[Action, int]] = []
    conflicts = 0
    for future in futures:
        try:
            successes.append(future.result())
        except DecisionConflict:
            conflicts += 1

    assert len(successes) == 1
    assert conflicts == 1
    assert successes[0][1] == 2
    stored = repository.get_snapshot("FC-DEMO-20418")
    assert stored is not None
    assert stored.version == 2
    assert stored.case.reviewer_decision is successes[0][0]
    assert len(repository.audit("FC-DEMO-20418")) == 1
    assert repository.verify_audit_chain("FC-DEMO-20418").valid is True


def test_stale_review_rolls_back_audit_and_case_update(tmp_path: Path) -> None:
    repository = CaseRepository(tmp_path / "rollback.db")
    case = synthetic_case()
    repository.upsert(case)
    recommendation = Copilot().recommend(case)
    Copilot.reviewer_decide(
        case,
        recommendation,
        action=Action.ESCALATE,
        reason="Synthetic stale reviewer decision.",
        reviewer_id="stale-reviewer",
    )

    with pytest.raises(DecisionConflict):
        repository.commit_review(
            case,
            known_audit_count=0,
            expected_version=99,
        )

    stored = repository.get_snapshot(case.case_id)
    assert stored is not None
    assert stored.version == 1
    assert stored.case.status == "open"
    assert repository.audit(case.case_id) == []
