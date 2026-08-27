from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from copilot import (
    Action,
    Evidence,
    EvidenceKind,
    FinancialCrimeCase,
    Severity,
    Signal,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decision_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    event_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_decision_audit_case_id ON decision_audit(case_id);
"""

GENESIS_AUDIT_HASH = "0" * 64


@dataclass(frozen=True)
class AuditVerification:
    valid: bool
    entries: int
    head_hash: str
    failure_index: int | None = None


def _evidence_from_dict(row: dict) -> Evidence:
    return Evidence(
        evidence_id=row["evidence_id"],
        kind=EvidenceKind(row["kind"]),
        source=row["source"],
        summary=row["summary"],
        confidence=float(row["confidence"]),
        event_time=row["event_time"],
        contradictory=bool(row.get("contradictory", False)),
        attributes=dict(row.get("attributes", {})),
    )


def _signal_from_dict(row: dict) -> Signal:
    return Signal(
        signal_id=row["signal_id"],
        name=row["name"],
        severity=Severity(row["severity"]),
        score=float(row["score"]),
        rationale=row["rationale"],
        evidence_ids=tuple(row.get("evidence_ids", [])),
        deterministic=bool(row.get("deterministic", True)),
    )


def _audit_hash(event: dict[str, object], previous_hash: str) -> str:
    canonical = json.dumps(
        {"event": event, "previous_hash": previous_hash},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _chain_event(event: dict[str, object], previous_hash: str) -> dict[str, object]:
    payload = dict(event)
    payload["previous_hash"] = previous_hash
    payload["event_hash"] = _audit_hash(event, previous_hash)
    return payload


def case_to_dict(case: FinancialCrimeCase) -> dict:
    payload = asdict(case)
    payload["customer_impact"] = case.customer_impact.value
    payload["reviewer_decision"] = (
        case.reviewer_decision.value if case.reviewer_decision is not None else None
    )
    for item in payload["evidence"]:
        item["kind"] = item["kind"].value if hasattr(item["kind"], "value") else item["kind"]
    for item in payload["signals"]:
        item["severity"] = (
            item["severity"].value if hasattr(item["severity"], "value") else item["severity"]
        )
    return payload


def case_from_dict(payload: dict) -> FinancialCrimeCase:
    decision = payload.get("reviewer_decision")
    return FinancialCrimeCase(
        case_id=payload["case_id"],
        subject_id=payload["subject_id"],
        subject_type=payload["subject_type"],
        opened_at=payload["opened_at"],
        evidence=[_evidence_from_dict(item) for item in payload.get("evidence", [])],
        signals=[_signal_from_dict(item) for item in payload.get("signals", [])],
        assigned_team=payload.get("assigned_team", "financial-crime-review"),
        sla_minutes=int(payload.get("sla_minutes", 60)),
        customer_impact=Severity(payload.get("customer_impact", Severity.MEDIUM.value)),
        status=payload.get("status", "open"),
        reviewer_decision=Action(decision) if decision else None,
        reviewer_reason=payload.get("reviewer_reason"),
        audit_events=list(payload.get("audit_events", [])),
    )


class CaseRepository:
    """SQLite-backed repository with a tamper-evident reviewer audit chain."""

    def __init__(self, database_path: str | Path = "financial_crime.db") -> None:
        self.database_path = str(database_path)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def upsert(self, case: FinancialCrimeCase) -> None:
        payload = json.dumps(case_to_dict(case), separators=(",", ":"), sort_keys=True)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cases(case_id, subject_id, status, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    subject_id = excluded.subject_id,
                    status = excluded.status,
                    payload = excluded.payload,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (case.case_id, case.subject_id, case.status, payload),
            )

    def get(self, case_id: str) -> FinancialCrimeCase | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        if row is None:
            return None
        return case_from_dict(json.loads(row["payload"]))

    def list(self, *, status: str | None = None, limit: int = 100) -> list[FinancialCrimeCase]:
        query = "SELECT payload FROM cases"
        params: list[object] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(limit, 500)))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [case_from_dict(json.loads(row["payload"])) for row in rows]

    def append_new_audit_events(self, case: FinancialCrimeCase, known_count: int) -> None:
        events = case.audit_events[known_count:]
        if not events:
            return

        with self._lock, self._connect() as connection:
            last_row = connection.execute(
                "SELECT event_json FROM decision_audit WHERE case_id = ? ORDER BY id DESC LIMIT 1",
                (case.case_id,),
            ).fetchone()
            if last_row is None:
                previous_hash = GENESIS_AUDIT_HASH
            else:
                last_event = json.loads(last_row["event_json"])
                previous_hash = str(last_event.get("event_hash", GENESIS_AUDIT_HASH))

            rows: list[tuple[str, str]] = []
            for event in events:
                chained = _chain_event(event, previous_hash)
                previous_hash = str(chained["event_hash"])
                rows.append(
                    (
                        case.case_id,
                        json.dumps(chained, sort_keys=True, separators=(",", ":"), default=str),
                    )
                )
            connection.executemany(
                "INSERT INTO decision_audit(case_id, event_json) VALUES (?, ?)",
                rows,
            )

    def audit(self, case_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT event_json FROM decision_audit WHERE case_id = ? ORDER BY id",
                (case_id,),
            ).fetchall()
        return [json.loads(row["event_json"]) for row in rows]

    def verify_audit_chain(self, case_id: str) -> AuditVerification:
        events = self.audit(case_id)
        previous_hash = GENESIS_AUDIT_HASH
        for index, stored in enumerate(events):
            claimed_previous = stored.get("previous_hash")
            claimed_hash = stored.get("event_hash")
            event = {
                key: value
                for key, value in stored.items()
                if key not in {"previous_hash", "event_hash"}
            }
            expected_hash = _audit_hash(event, previous_hash)
            if claimed_previous != previous_hash or claimed_hash != expected_hash:
                return AuditVerification(
                    valid=False,
                    entries=len(events),
                    head_hash=previous_hash,
                    failure_index=index,
                )
            previous_hash = str(claimed_hash)

        return AuditVerification(
            valid=True,
            entries=len(events),
            head_hash=previous_hash,
        )

    def seed(self, cases: Iterable[FinancialCrimeCase]) -> None:
        for case in cases:
            self.upsert(case)
