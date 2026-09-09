import threading
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from backend.constants import MODEL_TYPES
from backend.database import SessionLocal
from backend.model import Company, ModelRecord, TrainingJob
from backend.services.companies import ensure_model_records
from ml.model_paths import (
    PURPOSE_EVALUATION,
    PURPOSE_PRODUCTION,
    model_weights_path,
)
from ml.train_pipeline.train import main as admin_train_main
from ml.train_pipeline.user_train import main as user_train_main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_lock = threading.Lock()
_active_threads: dict[int, threading.Thread] = {}


def _utcnow():
    return datetime.now(timezone.utc)


def create_training_job(
    db: Session,
    company: Company,
    model_type: str,
    user_id: int | None,
    include_test: bool = True,
    epochs: int = 100,
) -> TrainingJob:
    """Queue training.

    include_test=True  -> evaluation model (held-out test from 2025)
    include_test=False -> production model (all data)
    """
    model_type = model_type.lower()
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model: {model_type}")

    active = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.company_id == company.id,
            TrainingJob.model_type == model_type,
            TrainingJob.include_test.is_(include_test),
            TrainingJob.status.in_(["queued", "training"]),
        )
        .first()
    )
    if active:
        purpose = "evaluation" if include_test else "production"
        raise ValueError(
            f"{purpose.title()} training already in progress for "
            f"{company.symbol} / {model_type}"
        )

    records = ensure_model_records(db, company)
    model_record = next(r for r in records if r.model_type == model_type)
    model_record.status = "training"
    db.commit()

    job = TrainingJob(
        company_id=company.id,
        model_id=model_record.id,
        user_id=user_id,
        model_type=model_type,
        status="queued",
        total_epochs=epochs,
        include_test=include_test,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def job_to_dict(job: TrainingJob) -> dict:
    purpose = PURPOSE_EVALUATION if job.include_test else PURPOSE_PRODUCTION
    return {
        "id": job.id,
        "company_id": job.company_id,
        "company": job.company.symbol if job.company else None,
        "model_type": job.model_type,
        "purpose": purpose,
        "status": job.status,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs,
        "train_loss": job.train_loss,
        "validation_loss": job.validation_loss,
        "error": job.error,
        "include_test": job.include_test,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }


def _run_job(job_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            return

        company = db.query(Company).filter(Company.id == job.company_id).first()
        model_record = (
            db.query(ModelRecord).filter(ModelRecord.id == job.model_id).first()
        )
        purpose = PURPOSE_EVALUATION if job.include_test else PURPOSE_PRODUCTION

        job.status = "training"
        job.started_at = _utcnow()
        job.error = None
        if model_record:
            model_record.status = "training"
        db.commit()

        def progress_callback(epoch, total_epochs, train_loss, val_loss):
            session = SessionLocal()
            try:
                j = session.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                if not j:
                    return
                j.current_epoch = epoch
                j.total_epochs = total_epochs
                j.train_loss = float(train_loss)
                j.validation_loss = float(val_loss) if val_loss is not None else None
                session.commit()
            finally:
                session.close()

        if job.include_test:
            admin_train_main(
                data=company.symbol,
                model_name=job.model_type,
                epochs=job.total_epochs or 100,
                include_test=True,
                purpose=PURPOSE_EVALUATION,
                progress_callback=progress_callback,
            )
        else:
            user_train_main(
                data=company.symbol,
                model_name=job.model_type,
                epochs=job.total_epochs or 100,
                progress_callback=progress_callback,
            )

        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        model_record = (
            db.query(ModelRecord).filter(ModelRecord.id == job.model_id).first()
        )
        company = db.query(Company).filter(Company.id == job.company_id).first()

        job.status = "completed"
        job.completed_at = _utcnow()
        job.current_epoch = job.total_epochs
        if model_record:
            weights = model_weights_path(company.symbol, job.model_type, purpose)
            if purpose == PURPOSE_EVALUATION:
                model_record.status = "trained"
                model_record.model_path = str(weights)
            else:
                # Production deploy completed; keep evaluated status if already set
                if model_record.status != "evaluated":
                    model_record.status = "trained"
            model_record.last_trained_at = _utcnow()
            model_record.version = (model_record.version or 0) + 1
            model_record.updated_at = _utcnow()
        db.commit()
    except Exception as exc:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error = str(exc)
            job.completed_at = _utcnow()
            if job.model_id:
                model_record = (
                    db.query(ModelRecord)
                    .filter(ModelRecord.id == job.model_id)
                    .first()
                )
                if model_record:
                    model_record.status = "failed"
            db.commit()
    finally:
        with _lock:
            _active_threads.pop(job_id, None)
        db.close()


def start_training_job(job_id: int) -> None:
    with _lock:
        if job_id in _active_threads and _active_threads[job_id].is_alive():
            return
        thread = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
        _active_threads[job_id] = thread
        thread.start()


def list_jobs(db: Session, limit: int = 50) -> list[dict]:
    jobs = (
        db.query(TrainingJob)
        .order_by(TrainingJob.id.desc())
        .limit(limit)
        .all()
    )
    return [job_to_dict(j) for j in jobs]
