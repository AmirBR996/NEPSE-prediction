from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.constants import MODEL_TYPES, ROLE_ADMIN, ROLE_USER
from backend.database import get_db
from backend.deps import require_admin, user_to_dict
from backend.model import (
    Company,
    ModelEvaluation,
    ModelRecord,
    TrainingJob,
    User,
)
from backend.services.companies import (
    company_overview,
    get_company_by_symbol,
    latest_evaluations,
    list_companies,
    model_file_exists,
    set_production_model,
)
from backend.services.prediction import run_company_evaluation
from backend.services.training import (
    create_training_job,
    job_to_dict,
    list_jobs,
    start_training_job,
)
from ml.model_paths import PURPOSE_EVALUATION, PURPOSE_PRODUCTION, TEST_SPLIT_DATE

router = APIRouter(prefix="/admin", tags=["Admin"])


class StatusUpdate(BaseModel):
    is_active: bool


class AuthorizeUpdate(BaseModel):
    is_authorized: bool


class TrainRequest(BaseModel):
    company: str
    model: str
    epochs: int = Field(default=100, ge=1, le=300)


class ProductionRequest(BaseModel):
    model_type: str
    epochs: int = Field(default=100, ge=1, le=300)


# ---------- Overview ----------


@router.get("/stats")
def admin_stats(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    total_users = db.query(User).filter(User.role != ROLE_ADMIN).count()
    authorized = (
        db.query(User)
        .filter(User.role != ROLE_ADMIN, User.is_authorized.is_(True))
        .count()
    )
    pending = (
        db.query(User)
        .filter(User.role != ROLE_ADMIN, User.is_authorized.is_(False))
        .count()
    )
    companies = db.query(Company).count()
    active_models = (
        db.query(ModelRecord).filter(ModelRecord.status.in_(["trained", "evaluated"])).count()
    )
    failed_jobs = db.query(TrainingJob).filter(TrainingJob.status == "failed").count()

    trained_companies = 0
    untrained_companies = 0
    for company in db.query(Company).all():
        trained_any = any(
            model_file_exists(company.symbol, m, PURPOSE_EVALUATION)
            for m in MODEL_TYPES
        )
        if trained_any:
            trained_companies += 1
        else:
            untrained_companies += 1

    return {
        "users": {
            "total": total_users,
            "authorized": authorized,
            "pending": pending,
        },
        "companies": {
            "total": companies,
            "trained": trained_companies,
            "untrained": untrained_companies,
        },
        "active_models": active_models,
        "failed_training_jobs": failed_jobs,
        "test_split_date": TEST_SPLIT_DATE,
        "notes": {
            "evaluation": f"Trains with held-out test from {TEST_SPLIT_DATE}",
            "production": "Trains on all data; used for live user predictions",
        },
    }


@router.get("/system")
def system_status(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    active_jobs = (
        db.query(TrainingJob)
        .filter(TrainingJob.status.in_(["queued", "training"]))
        .count()
    )
    return {
        "api": "ok",
        "database": "ok",
        "active_training_jobs": active_jobs,
        "companies": db.query(Company).count(),
        "models_tracked": db.query(ModelRecord).count(),
        "evaluations": db.query(ModelEvaluation).count(),
    }


# ---------- Users ----------


@router.get("/users")
def list_users(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    users = db.query(User).order_by(User.id.asc()).all()
    return {"users": [user_to_dict(u) for u in users]}


@router.patch("/users/{user_id}/authorize")
def authorize_user(
    user_id: int,
    payload: AuthorizeUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if (user.role or ROLE_USER) == ROLE_ADMIN:
        raise HTTPException(status_code=400, detail="Admin authorization cannot be changed")

    user.is_authorized = payload.is_authorized
    db.commit()
    db.refresh(user)
    return {"message": "Authorization updated", "user": user_to_dict(user)}


@router.patch("/users/{user_id}/status")
def update_user_status(
    user_id: int,
    payload: StatusUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id and not payload.is_active:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")

    if (user.role or ROLE_USER) == ROLE_ADMIN and not payload.is_active:
        admin_count = (
            db.query(User)
            .filter(User.role == ROLE_ADMIN, User.is_active.is_(True))
            .count()
        )
        if admin_count <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot deactivate the final active admin account",
            )

    user.is_active = payload.is_active
    db.commit()
    db.refresh(user)
    return {"message": "Status updated", "user": user_to_dict(user)}


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
    confirm: bool = Query(False),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    if (user.role or ROLE_USER) == ROLE_ADMIN:
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Deleting an admin requires confirm=true",
            )
        admin_count = db.query(User).filter(User.role == ROLE_ADMIN).count()
        if admin_count <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete the final admin account",
            )

    db.delete(user)
    db.commit()
    return {"message": "User deleted", "id": user_id}


# ---------- Companies ----------


@router.get("/companies")
def admin_companies(
    status: str | None = Query(None),
    q: str | None = Query(None),
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    companies = list_companies(db)
    if q:
        ql = q.lower()
        companies = [
            c
            for c in companies
            if ql in c["symbol"].lower() or ql in (c.get("name") or "").lower()
        ]

    def matches(c):
        if not status or status.lower() == "all":
            return True
        s = status.lower()
        ts = (c.get("training_status") or "").lower()
        trained_flags = [
            c["models"][m]["has_evaluation_weights"]
            for m in MODEL_TYPES
            if m in c.get("models", {})
        ]
        if s == "trained":
            return any(trained_flags)
        if s == "untrained":
            return not any(trained_flags)
        if s == "training":
            return "training" in ts
        if s == "failed":
            return any(
                c["models"][m]["status"] == "failed"
                for m in c.get("models", {})
            )
        if s == "evaluated":
            return any(
                c["models"][m]["status"] == "evaluated"
                for m in c.get("models", {})
            )
        if s in ("not_evaluated", "not evaluated"):
            return any(trained_flags) and not any(
                c["models"][m]["status"] == "evaluated"
                for m in c.get("models", {})
            )
        return True

    return {"companies": [c for c in companies if matches(c)]}


@router.post("/companies/{company_id}/train")
def admin_train_company(
    company_id: str,
    payload: TrainRequest | None = None,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    company = _resolve(db, company_id)
    model = (payload.model if payload else None) or "lstm"
    epochs = payload.epochs if payload else 100
    model = model.lower()
    if model not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    try:
        job = create_training_job(
            db=db,
            company=company,
            model_type=model,
            user_id=admin.id,
            include_test=True,
            epochs=epochs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    start_training_job(job.id)
    return {"message": "Training queued", "job": job_to_dict(job)}


@router.post("/companies/{company_id}/evaluate")
def admin_evaluate_company(
    company_id: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    company = _resolve(db, company_id)
    try:
        return run_company_evaluation(db, company)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------- Models ----------


@router.get("/models")
def admin_models(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    payload = []
    for company in db.query(Company).order_by(Company.symbol).all():
        overview = company_overview(db, company)
        payload.append(
            {
                "company": company.symbol,
                "name": company.name,
                "production_model": company.production_model,
                "status": overview["training_status"],
                "models": overview["models"],
            }
        )
    return {"models": payload}


@router.get("/models/{company_id}")
def admin_company_models(
    company_id: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    company = _resolve(db, company_id)
    overview = company_overview(db, company)
    return {
        "company": company.symbol,
        "production_model": company.production_model,
        "models": overview["models"],
        "evaluations": latest_evaluations(db, company),
    }


@router.post("/models/{model_id}/production")
def set_model_production(
    model_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    record = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Model not found")
    company = db.query(Company).filter(Company.id == record.company_id).first()
    try:
        set_production_model(db, company, record.model_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    deploy_job = None
    try:
        # Retrain selected architecture on ALL data for live predictions
        job = create_training_job(
            db=db,
            company=company,
            model_type=record.model_type,
            user_id=admin.id,
            include_test=False,
            epochs=100,
        )
        start_training_job(job.id)
        deploy_job = job_to_dict(job)
    except ValueError as exc:
        # Already deploying is OK; selection still saved
        deploy_job = {"warning": str(exc)}

    return {
        "message": (
            f"{record.model_type} selected as production for {company.symbol}. "
            "Production training on all data has been queued."
        ),
        "company": company.symbol,
        "production_model": company.production_model,
        "deploy_job": deploy_job,
    }


@router.post("/companies/{company_id}/production")
def set_company_production(
    company_id: str,
    payload: ProductionRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    company = _resolve(db, company_id)
    try:
        set_production_model(db, company, payload.model_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    epochs = getattr(payload, "epochs", None) or 100
    deploy_job = None
    try:
        job = create_training_job(
            db=db,
            company=company,
            model_type=payload.model_type.lower(),
            user_id=admin.id,
            include_test=False,
            epochs=epochs,
        )
        start_training_job(job.id)
        deploy_job = job_to_dict(job)
    except ValueError as exc:
        deploy_job = {"warning": str(exc)}

    return {
        "message": (
            f"{payload.model_type} selected as production for {company.symbol}. "
            "Production training on all data has been queued."
        ),
        "company": company.symbol,
        "production_model": company.production_model,
        "deploy_job": deploy_job,
    }


# ---------- Evaluation ----------


@router.get("/evaluations/{company_id}")
def get_evaluations(
    company_id: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    company = _resolve(db, company_id)
    return {
        "company": company.symbol,
        "production_model": company.production_model,
        "evaluations": latest_evaluations(db, company),
    }


@router.get("/evaluations/{company_id}/{model_type}")
def get_evaluation_model(
    company_id: str,
    model_type: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    company = _resolve(db, company_id)
    model_type = model_type.lower()
    evaluations = latest_evaluations(db, company)
    match = next((e for e in evaluations if e["model"] == model_type), None)
    if not match:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return {"company": company.symbol, **match}


@router.post("/evaluate/{data}")
def evaluate_company_legacy(
    data: str,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    company = get_company_by_symbol(db, data)
    if not company:
        raise HTTPException(status_code=400, detail=f"Unsupported company: {data}")
    try:
        return run_company_evaluation(db, company)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------- Training ----------


@router.get("/training/jobs")
def admin_training_jobs(
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    return {"jobs": list_jobs(db)}


@router.post("/training")
def admin_start_training(
    payload: TrainRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    company = get_company_by_symbol(db, payload.company)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    model = payload.model.lower()
    if model not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

    try:
        job = create_training_job(
            db=db,
            company=company,
            model_type=model,
            user_id=admin.id,
            include_test=True,
            epochs=payload.epochs,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    start_training_job(job.id)
    return {"message": "Training queued", "job": job_to_dict(job)}


@router.post("/trainall")
def train_all_admin(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    queued = []
    errors = []
    for company in db.query(Company).all():
        for model_name in MODEL_TYPES:
            try:
                job = create_training_job(
                    db=db,
                    company=company,
                    model_type=model_name,
                    user_id=admin.id,
                    include_test=True,
                    epochs=100,
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
        "message": "Training jobs queued",
        "queued": queued,
        "errors": errors,
    }


@router.post("/{model}/{data}")
def train_one_legacy(
    model: str,
    data: str,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    if model == "evaluate":
        raise HTTPException(status_code=404, detail="Use /admin/evaluate/{data}")
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
            user_id=admin.id,
            include_test=True,
            epochs=100,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    start_training_job(job.id)
    return {
        "response": f"{model} training queued for {data}",
        "job": job_to_dict(job),
    }


@router.post("/{data}")
def train_all_for_company_legacy(
    data: str,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    # Avoid swallowing dedicated paths
    reserved = {
        "trainall",
        "users",
        "companies",
        "models",
        "stats",
        "system",
        "training",
        "evaluations",
        "evaluate",
    }
    if data in reserved:
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
                user_id=admin.id,
                include_test=True,
                epochs=100,
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


def _resolve(db: Session, company_id: str) -> Company:
    company = None
    if str(company_id).isdigit():
        company = db.query(Company).filter(Company.id == int(company_id)).first()
    if not company:
        company = get_company_by_symbol(db, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return company
