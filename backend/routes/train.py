from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.constants import MODEL_TYPES
from backend.database import get_db
from backend.deps import require_authorized_user
from backend.model import User
from backend.services.companies import get_company_by_symbol
from backend.services.training import (
    create_training_job,
    job_to_dict,
    start_training_job,
)

router = APIRouter(prefix="/train", tags=["train"])


@router.post("/trainall")
def train_all(
    db: Session = Depends(get_db),
    user: User = Depends(require_authorized_user),
):
    queued = []
    errors = []
    from backend.model import Company

    for company in db.query(Company).filter(Company.is_active.is_(True)).all():
        for model_name in MODEL_TYPES:
            try:
                job = create_training_job(
                    db=db,
                    company=company,
                    model_type=model_name,
                    user_id=user.id,
                    include_test=False,
                    epochs=50,
                )
                start_training_job(job.id)
                queued.append(job_to_dict(job))
            except ValueError as exc:
                errors.append(
                    {
                        "company": company.symbol,
                        "model": model_name,
                        "error": str(exc),
                    }
                )
    return {
        "message": "User training jobs queued",
        "queued": queued,
        "errors": errors,
    }


@router.post("/{model}/{data}")
def train_one(
    model: str,
    data: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_authorized_user),
):
    if model not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    company = get_company_by_symbol(db, data)
    if not company:
        raise HTTPException(status_code=400, detail=f"Unsupported company: {data}")

    try:
        job = create_training_job(
            db=db,
            company=company,
            model_type=model,
            user_id=user.id,
            include_test=False,
            epochs=50,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    start_training_job(job.id)
    return {
        "response": f"{model} training queued for {data}",
        "job": job_to_dict(job),
    }


@router.post("/{data}")
def train_company_all(
    data: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_authorized_user),
):
    if data == "trainall":
        raise HTTPException(status_code=404, detail="Not found")

    company = get_company_by_symbol(db, data)
    if not company:
        raise HTTPException(status_code=400, detail=f"Unsupported company: {data}")

    results = []
    for model_name in MODEL_TYPES:
        try:
            job = create_training_job(
                db=db,
                company=company,
                model_type=model_name,
                user_id=user.id,
                include_test=False,
                epochs=50,
            )
            start_training_job(job.id)
            results.append(job_to_dict(job))
        except ValueError as exc:
            results.append(
                {"company": data, "model": model_name, "error": str(exc)}
            )

    return {
        "response": f"Training queued for all models on {data}",
        "results": results,
    }
