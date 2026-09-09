import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from backend.constants import COMPANY_NAMES, COMPANY_SYMBOLS, MODEL_TYPES
from backend.model import Company, ModelEvaluation, ModelRecord, TrainingJob
from ml.model_paths import (
    PURPOSE_EVALUATION,
    PURPOSE_PRODUCTION,
    TEST_SPLIT_DATE,
    model_exists,
    model_weights_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def get_company_by_symbol(db: Session, symbol: str) -> Company | None:
    return db.query(Company).filter(Company.symbol == symbol.upper()).first()


def ensure_model_records(db: Session, company: Company) -> list[ModelRecord]:
    records = []
    for model_type in MODEL_TYPES:
        record = (
            db.query(ModelRecord)
            .filter(
                ModelRecord.company_id == company.id,
                ModelRecord.model_type == model_type,
            )
            .first()
        )
        if not record:
            record = ModelRecord(
                company_id=company.id,
                model_type=model_type,
                status="not_trained",
            )
            db.add(record)
            db.flush()
        records.append(record)
    return records


def model_file_exists(
    symbol: str,
    model_type: str,
    purpose: str = PURPOSE_EVALUATION,
) -> bool:
    return model_exists(symbol, model_type, purpose)


def sync_model_status_from_disk(db: Session, company: Company) -> None:
    for record in ensure_model_records(db, company):
        has_eval = model_file_exists(
            company.symbol, record.model_type, PURPOSE_EVALUATION
        )
        has_prod = model_file_exists(
            company.symbol, record.model_type, PURPOSE_PRODUCTION
        )
        if has_eval and record.status in (None, "not_trained", "failed"):
            record.status = "trained"
        if has_eval and not record.model_path:
            record.model_path = str(
                model_weights_path(
                    company.symbol, record.model_type, PURPOSE_EVALUATION
                )
            )
        if has_prod and company.production_model == record.model_type:
            record.is_production = True
    db.commit()


def load_company_csv(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data not found for {symbol}")
    df = pd.read_csv(path)
    df["published_date"] = pd.to_datetime(df["published_date"])
    df = df.sort_values("published_date").reset_index(drop=True)
    return df


def company_overview(db: Session, company: Company) -> dict:
    sync_model_status_from_disk(db, company)
    overview = {
        "id": company.id,
        "symbol": company.symbol,
        "name": company.name or COMPANY_NAMES.get(company.symbol, company.symbol),
        "is_active": company.is_active,
        "production_model": company.production_model,
        "test_split_date": TEST_SPLIT_DATE,
        "latest_price": None,
        "previous_price": None,
        "percentage_change": None,
        "trading_volume": None,
        "last_updated": None,
        "training_status": "Not Trained",
        "models": {},
    }

    try:
        df = load_company_csv(company.symbol)
        if len(df) >= 1:
            latest = df.iloc[-1]
            overview["latest_price"] = float(latest["close"])
            overview["trading_volume"] = float(latest.get("traded_quantity") or 0)
            overview["last_updated"] = latest["published_date"].strftime("%Y-%m-%d")
            if "per_change" in df.columns and pd.notna(latest.get("per_change")):
                overview["percentage_change"] = float(latest["per_change"])
        if len(df) >= 2:
            prev = df.iloc[-2]
            overview["previous_price"] = float(prev["close"])
            if overview["percentage_change"] is None and overview["previous_price"]:
                overview["percentage_change"] = (
                    (overview["latest_price"] - overview["previous_price"])
                    / overview["previous_price"]
                ) * 100
    except FileNotFoundError:
        pass

    eval_count = 0
    prod_ready = False
    for record in ensure_model_records(db, company):
        status = record.status or "not_trained"
        has_eval = model_file_exists(
            company.symbol, record.model_type, PURPOSE_EVALUATION
        )
        has_prod = model_file_exists(
            company.symbol, record.model_type, PURPOSE_PRODUCTION
        )
        if has_eval:
            eval_count += 1
            if status == "not_trained":
                status = "trained"
        is_prod = bool(record.is_production) or company.production_model == record.model_type
        if is_prod and has_prod:
            prod_ready = True
        overview["models"][record.model_type] = {
            "id": record.id,
            "status": status,
            "is_production": is_prod,
            "last_trained_at": record.last_trained_at.isoformat()
            if record.last_trained_at
            else None,
            "last_evaluated_at": record.last_evaluated_at.isoformat()
            if record.last_evaluated_at
            else None,
            "has_weights": has_eval,
            "has_evaluation_weights": has_eval,
            "has_production_weights": has_prod,
        }

    active_job = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.company_id == company.id,
            TrainingJob.status.in_(["queued", "training"]),
        )
        .order_by(TrainingJob.id.desc())
        .first()
    )
    if active_job:
        purpose = "Evaluation" if active_job.include_test else "Production"
        overview["training_status"] = f"Training ({purpose})"
    elif eval_count == 0:
        overview["training_status"] = "Not Trained"
    elif prod_ready:
        overview["training_status"] = "Production Ready"
    elif company.production_model:
        overview["training_status"] = "Production Deploy Pending"
    else:
        eval_rows = (
            db.query(ModelEvaluation)
            .filter(ModelEvaluation.company_id == company.id)
            .count()
        )
        if eval_rows > 0:
            overview["training_status"] = "Evaluated"
        elif eval_count < len(MODEL_TYPES):
            overview["training_status"] = "Partially Trained"
        else:
            overview["training_status"] = "Trained"

    return overview


def company_history(symbol: str, range_key: str = "1Y") -> dict:
    df = load_company_csv(symbol)
    if df.empty:
        return {"symbol": symbol, "range": range_key, "points": []}

    end = df["published_date"].max()
    range_map = {
        "1M": pd.DateOffset(months=1),
        "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6),
        "1Y": pd.DateOffset(years=1),
        "5Y": pd.DateOffset(years=5),
        "ALL": None,
    }
    offset = range_map.get(range_key.upper(), range_map["1Y"])
    filtered = df if offset is None else df[df["published_date"] >= (end - offset)]

    points = []
    for _, row in filtered.iterrows():
        points.append(
            {
                "date": row["published_date"].strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("traded_quantity") or 0),
                "per_change": float(row["per_change"])
                if pd.notna(row.get("per_change"))
                else None,
            }
        )
    return {"symbol": symbol.upper(), "range": range_key.upper(), "points": points}


def list_companies(db: Session) -> list[dict]:
    companies = db.query(Company).order_by(Company.symbol).all()
    if not companies:
        for symbol in COMPANY_SYMBOLS:
            companies.append(
                Company(symbol=symbol, name=COMPANY_NAMES.get(symbol, symbol))
            )
        db.add_all(companies)
        db.commit()
        for c in companies:
            db.refresh(c)

    return [company_overview(db, c) for c in companies]


def set_production_model(db: Session, company: Company, model_type: str) -> Company:
    """Select architecture for production after evaluation comparison.

    Evaluation weights stay untouched. Production weights are trained separately
    on all data (deploy job).
    """
    model_type = model_type.lower()
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model: {model_type}")

    records = ensure_model_records(db, company)
    target = next((r for r in records if r.model_type == model_type), None)
    if not target or not model_file_exists(
        company.symbol, model_type, PURPOSE_EVALUATION
    ):
        raise ValueError(
            f"No {model_type} evaluation model for {company.symbol}. "
            "Train with the 2025 hold-out split first."
        )

    for record in records:
        record.is_production = record.model_type == model_type
        record.updated_at = datetime.now(timezone.utc)

    company.production_model = model_type
    db.commit()
    db.refresh(company)
    return company


def latest_evaluations(db: Session, company: Company) -> list[dict]:
    results = []
    for model_type in MODEL_TYPES:
        evaluation = (
            db.query(ModelEvaluation)
            .filter(
                ModelEvaluation.company_id == company.id,
                ModelEvaluation.model_type == model_type,
            )
            .order_by(ModelEvaluation.evaluation_date.desc())
            .first()
        )
        record = (
            db.query(ModelRecord)
            .filter(
                ModelRecord.company_id == company.id,
                ModelRecord.model_type == model_type,
            )
            .first()
        )
        has_eval = model_file_exists(company.symbol, model_type, PURPOSE_EVALUATION)
        has_prod = model_file_exists(company.symbol, model_type, PURPOSE_PRODUCTION)
        item = {
            "model": model_type,
            "has_weights": has_eval,
            "has_evaluation_weights": has_eval,
            "has_production_weights": has_prod,
            "is_production": bool(record and record.is_production)
            or company.production_model == model_type,
            "status": record.status if record else "not_trained",
            "metrics": None,
            "series": None,
            "test_split_date": TEST_SPLIT_DATE,
        }
        if evaluation:
            item["metrics"] = {
                "mae": evaluation.mae,
                "rmse": evaluation.rmse,
                "mape": evaluation.mape,
                "r2": evaluation.r2,
                "directional_accuracy": evaluation.directional_accuracy,
            }
            if evaluation.series_json:
                try:
                    item["series"] = json.loads(evaluation.series_json)
                except json.JSONDecodeError:
                    item["series"] = None
            item["evaluation_date"] = (
                evaluation.evaluation_date.isoformat()
                if evaluation.evaluation_date
                else None
            )
        results.append(item)
    return results
