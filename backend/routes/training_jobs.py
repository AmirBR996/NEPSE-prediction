from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.constants import MODEL_TYPES
from backend.database import get_db
from backend.deps import require_authorized_user, require_user
from backend.model import TrainingJob, User
from backend.services.companies import get_company_by_symbol
from backend.services.training import (
    create_training_job,
    job_to_dict,
    start_training_job,
)

router = APIRouter(prefix="/training", tags=["Training"])


class UserTrainRequest(BaseModel):
    company: str
    model: str
    epochs: int = Field(default=50, ge=1, le=200)


@router.post("/user")
def start_user_training(
    payload: UserTrainRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_authorized_user),
):
    company = get_company_by_symbol(db, payload.company)
    if not company or not company.is_active:
        raise HTTPException(status_code=404, detail="Company not found")

    model = payload.model.lower()
    if model not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    try:
        job = create_training_job(
            db=db,
            company=company,
            model_type=model,
            user_id=user.id,
            include_test=False,
            epochs=payload.epochs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    start_training_job(job.id)
    return {"message": "Training queued", "job": job_to_dict(job)}


@router.get("/{job_id}")
def get_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    role = user.role or ("ADMIN" if user.is_admin else "USER")
    if role != "ADMIN" and job.user_id not in (None, user.id):
        raise HTTPException(status_code=403, detail="Not allowed to view this job")

    return job_to_dict(job)
