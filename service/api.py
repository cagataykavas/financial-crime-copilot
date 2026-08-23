from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from copilot import Action, Copilot, synthetic_case
from service.store import CaseRepository, case_to_dict


class ReviewerDecisionRequest(BaseModel):
    action: Action
    reason: str = Field(min_length=5, max_length=1000)
    reviewer_id: str = Field(min_length=2, max_length=120)


DATABASE_PATH = Path(os.getenv("FC_DATABASE_PATH", "financial_crime.db"))
repository = CaseRepository(DATABASE_PATH)
copilot = Copilot()
app = FastAPI(
    title="Financial Crime Copilot",
    version="0.2.0",
    description=(
        "Synthetic human-in-the-loop financial-crime decision support API. "
        "Material dispositions remain reviewer-authorized."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        repository.list(limit=1)
    except Exception as exc:  # pragma: no cover - operational guard
        raise HTTPException(status_code=503, detail=f"repository unavailable: {exc}") from exc
    return {"status": "ready"}


@app.post("/demo/seed", status_code=201)
def seed_demo_case() -> dict:
    case = synthetic_case()
    repository.upsert(case)
    return case_to_dict(case)


@app.get("/cases")
def list_cases(
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict]:
    return [case_to_dict(case) for case in repository.list(status=status, limit=limit)]


@app.get("/cases/{case_id}")
def get_case(case_id: str) -> dict:
    case = repository.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    return case_to_dict(case)


@app.get("/cases/{case_id}/recommendation")
def get_recommendation(case_id: str) -> dict:
    case = repository.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    return asdict(copilot.recommend(case))


@app.post("/cases/{case_id}/decision")
def reviewer_decision(case_id: str, request: ReviewerDecisionRequest) -> dict:
    case = repository.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    if case.status == "resolved":
        raise HTTPException(status_code=409, detail="case already resolved")

    recommendation = copilot.recommend(case)
    known_audit_count = len(case.audit_events)
    copilot.reviewer_decide(
        case,
        recommendation,
        action=request.action,
        reason=request.reason,
        reviewer_id=request.reviewer_id,
    )
    repository.append_new_audit_events(case, known_audit_count)
    repository.upsert(case)
    return {
        "case": case_to_dict(case),
        "recommendation": asdict(recommendation),
        "override": request.action != recommendation.recommended_action,
    }


@app.get("/cases/{case_id}/audit")
def audit(case_id: str) -> list[dict]:
    if repository.get(case_id) is None:
        raise HTTPException(status_code=404, detail="case not found")
    return repository.audit(case_id)
