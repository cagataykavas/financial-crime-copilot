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
    listed = client.get("/cases").json()
    assert listed[0]["case_id"] == case_id
    assert listed[0]["version"] == 1

    recommendation = client.get(f"/cases/{case_id}/recommendation")
    assert recommendation.status_code == 200
    assert recommendation.json()["recommended_action"] == "escalate"

    policy = client.get(f"/cases/{case_id}/policy")
    assert policy.status_code == 200
    assert policy.json()["outcome"] == "require_human"
    assert "material_disposition_requires_human_authorization" in policy.json()["reasons"]

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
    assert decision.json()["policy"]["outcome"] == "require_human"

    audit = client.get(f"/cases/{case_id}/audit")
    assert audit.status_code == 200
    assert len(audit.json()) == 1
    assert audit.json()[0]["reviewer_id"] == "reviewer-test"
    assert len(audit.json()[0]["event_hash"]) == 64

    verification = client.get(f"/cases/{case_id}/audit/verify")
    assert verification.status_code == 200
    assert verification.json()["valid"] is True
    assert verification.json()["entries"] == 1


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


def test_stale_explicit_case_version_is_rejected_without_audit(tmp_path: Path) -> None:
    api_module.repository = CaseRepository(tmp_path / "stale.db")
    client = TestClient(api_module.app)
    case_id = client.post("/demo/seed").json()["case_id"]

    response = client.post(
        f"/cases/{case_id}/decision",
        json={
            "action": "escalate",
            "reason": "This reviewer screen contains a stale case version.",
            "reviewer_id": "reviewer-stale",
            "expected_version": 99,
        },
    )

    assert response.status_code == 409
    assert client.get(f"/cases/{case_id}/audit").json() == []
    current = client.get(f"/cases/{case_id}").json()
    assert current["status"] == "open"
    assert current["version"] == 1
