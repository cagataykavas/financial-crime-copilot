from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from copilot import Action, Copilot, synthetic_case
from service.store import CaseRepository, GENESIS_AUDIT_HASH


def _reviewed_case():
    case = synthetic_case()
    copilot = Copilot()
    recommendation = copilot.recommend(case)
    copilot.reviewer_decide(
        case,
        recommendation,
        action=Action.ESCALATE,
        reason="Synthetic review event for audit-chain validation.",
        reviewer_id="audit-reviewer",
    )
    return case


def test_audit_chain_is_valid_for_untampered_events(tmp_path: Path) -> None:
    repository = CaseRepository(tmp_path / "audit.db")
    case = _reviewed_case()
    repository.upsert(case)
    repository.append_new_audit_events(case, known_count=0)

    events = repository.audit(case.case_id)
    verification = repository.verify_audit_chain(case.case_id)

    assert len(events) == 1
    assert events[0]["previous_hash"] == GENESIS_AUDIT_HASH
    assert len(events[0]["event_hash"]) == 64
    assert verification.valid is True
    assert verification.entries == 1
    assert verification.head_hash == events[0]["event_hash"]
    assert verification.failure_index is None


def test_audit_chain_detects_database_tampering(tmp_path: Path) -> None:
    database = tmp_path / "audit.db"
    repository = CaseRepository(database)
    case = _reviewed_case()
    repository.upsert(case)
    repository.append_new_audit_events(case, known_count=0)

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT id, event_json FROM decision_audit WHERE case_id = ?",
            (case.case_id,),
        ).fetchone()
        assert row is not None
        event = json.loads(row[1])
        event["reason"] = "tampered-after-review"
        connection.execute(
            "UPDATE decision_audit SET event_json = ? WHERE id = ?",
            (json.dumps(event, sort_keys=True), row[0]),
        )

    verification = repository.verify_audit_chain(case.case_id)
    assert verification.valid is False
    assert verification.failure_index == 0
