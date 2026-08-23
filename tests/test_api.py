from pathlib import Path

from fastapi.testclient import TestClient

import service.api as api_module
from service.store import CaseRepository


def test_end_to_end_review_flow(tmp_path: Path) -> None:
    api_module.repository = CaseRepository(tmp_path / "test.db")
    client = TestClient(api_module.app)

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200

    seeded = client.post("/demo/seed")
    assert seeded.status_code == 201
    case_id = seeded.json()["case_id"]

    recommendation = client.get(f"/cases/{case_id}/recommendation")
    assert recommendation.status_code == 200
    assert recommendation.json()["recommended_action"] == "escalate"

    decision = client.post(
        f"/cases/{case_id}/decision",
        json={
            "action": "escalate",
            "reason": "Synthetic pattern requires specialist review.",
            "reviewer_id": "reviewer-test",
        },
    )
    assert decision.status_code == 200
    assert decision.json()["override"] is False
    assert decision.json()["case"]["status"] == "resolved"

    audit = client.get(f"/cases/{case_id}/audit")
    assert audit.status_code == 200
    assert len(audit.json()) == 1
    assert audit.json()[0]["reviewer_id"] == "reviewer-test"


def test_duplicate_decision_is_rejected(tmp_path: Path) -> None:
    api_module.repository = CaseRepository(tmp_path / "test.db")
    client = TestClient(api_module.app)
    case_id = client.post("/demo/seed").json()["case_id"]
    payload = {
        "action": "escalate",
        "reason": "First valid review decision.",
        "reviewer_id": "reviewer-test",
    }
    assert client.post(f"/cases/{case_id}/decision", json=payload).status_code == 200
    assert client.post(f"/cases/{case_id}/decision", json=payload).status_code == 409
