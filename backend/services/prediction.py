import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sqlalchemy.orm import Session

from backend.model import Company, ModelEvaluation, Prediction
from backend.services.companies import (
    ensure_model_records,
    model_file_exists,
)
from ml.data_preprocessing.feature import features
from ml.model_evaluation.evaluation import evaluate, load_model
from ml.model_paths import (
    PURPOSE_EVALUATION,
    PURPOSE_PRODUCTION,
    resolve_scaler_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_COLS = [
    "open",
    "high",
    "low",
    "traded_quantity",
    "traded_amount",
    "return_1",
    "return_5",
    "return_10",
]
SEQ_LENGTH = 30


def persist_evaluation(
    db: Session,
    company: Company,
    eval_result: dict,
) -> ModelEvaluation:
    model_type = eval_result["model"]
    metrics = eval_result["metrics"]
    records = ensure_model_records(db, company)
    model_record = next(r for r in records if r.model_type == model_type)

    series_payload = None
    if eval_result.get("series"):
        series_payload = json.dumps(eval_result["series"])

    evaluation = ModelEvaluation(
        company_id=company.id,
        model_id=model_record.id,
        model_type=model_type,
        mae=metrics.get("mae"),
        rmse=metrics.get("rmse"),
        mape=metrics.get("mape"),
        r2=metrics.get("r2"),
        directional_accuracy=metrics.get("directional_accuracy"),
        evaluation_date=datetime.now(timezone.utc),
        series_json=series_payload,
    )
    db.add(evaluation)
    model_record.status = "evaluated"
    model_record.last_evaluated_at = datetime.now(timezone.utc)
    model_record.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(evaluation)
    return evaluation


def run_company_evaluation(db: Session, company: Company) -> dict:
    """Score evaluation-split models on the held-out test period (from 2025)."""
    results = []
    errors = []
    for model_type in ("lstm", "gru", "transformer"):
        if not model_file_exists(company.symbol, model_type, PURPOSE_EVALUATION):
            errors.append(
                {
                    "model": model_type,
                    "error": "Evaluation model weights not found. Train evaluation models first.",
                }
            )
            continue
        try:
            result = evaluate(
                data=company.symbol,
                model_name=model_type,
                include_series=True,
            )
            persist_evaluation(db, company, result)
            results.append(
                {
                    "model": model_type,
                    "metrics": result["metrics"],
                    "series": result.get("series"),
                    "purpose": PURPOSE_EVALUATION,
                    "test_split_date": result.get("test_split_date"),
                }
            )
        except Exception as exc:
            errors.append({"model": model_type, "error": str(exc)})

    return {"company": company.symbol, "results": results, "errors": errors}


def _build_latest_sequence(symbol: str):
    """Inference features + production scalers fitted on all data."""
    csv_path = PROJECT_ROOT / "data" / f"{symbol}.csv"
    df = pd.read_csv(csv_path)
    df = features(df)
    if len(df) < SEQ_LENGTH + 1:
        raise ValueError("Insufficient history for prediction")

    scaler_x_path, scaler_y_path = resolve_scaler_paths(symbol, PURPOSE_PRODUCTION)
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    recent = df.iloc[-SEQ_LENGTH:].copy()
    x_raw = recent[FEATURE_COLS].values
    x_scaled = scaler_x.transform(x_raw)
    x_tensor = torch.tensor(x_scaled[np.newaxis, :, :], dtype=torch.float32)
    current_price = float(df.iloc[-1]["close"])
    return x_tensor, scaler_y, current_price, df


def predict_next_days(
    db: Session,
    company: Company,
    user_id: int,
    model_type: str | None = None,
    horizon_days: int = 1,
) -> dict:
    """Live predictions always use production (all-data) weights."""
    horizon_days = max(1, min(int(horizon_days), 7))
    records = ensure_model_records(db, company)

    selected = model_type.lower() if model_type else company.production_model
    if not selected:
        prod = next((r for r in records if r.is_production), None)
        selected = prod.model_type if prod else None
    if not selected:
        raise ValueError(
            "No production model configured for this company. "
            "Admin must evaluate models and deploy a production model."
        )
    if not model_file_exists(company.symbol, selected, PURPOSE_PRODUCTION):
        raise ValueError(
            f"Production weights not found for {selected}. "
            "Admin must deploy/train the production model on all data."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor, scaler_y, current_price, df = _build_latest_sequence(company.symbol)
    input_dim = x_tensor.shape[-1]
    model = load_model(
        company.symbol,
        selected,
        input_dim,
        device,
        purpose=PURPOSE_PRODUCTION,
    )

    predictions = []
    price = current_price
    seq = x_tensor.clone()

    model.eval()
    with torch.no_grad():
        for day in range(1, horizon_days + 1):
            pred_scaled = model(seq.to(device)).cpu().numpy()
            pred_return = float(scaler_y.inverse_transform(pred_scaled).flatten()[0])
            pred_return = float(np.clip(pred_return, -0.2, 0.2))
            next_price = price * (1.0 + pred_return)
            predictions.append(
                {
                    "day": day,
                    "predicted_return": pred_return,
                    "predicted_price": next_price,
                }
            )

            last_row = seq[0, -1].cpu().numpy().copy()
            last_row[5] = pred_return
            new_row = torch.tensor(last_row, dtype=torch.float32).unsqueeze(0)
            seq = torch.cat([seq[:, 1:, :], new_row.unsqueeze(0)], dim=1)
            price = next_price

    final_price = predictions[-1]["predicted_price"]
    change_pct = ((final_price - current_price) / current_price) * 100

    result = {
        "company": company.symbol,
        "model": selected,
        "purpose": PURPOSE_PRODUCTION,
        "horizon_days": horizon_days,
        "current_price": current_price,
        "predicted_price": final_price,
        "predicted_change_pct": change_pct,
        "predictions": predictions,
        "last_updated": df.iloc[-1]["published_date"].strftime("%Y-%m-%d")
        if "published_date" in df.columns
        else None,
    }

    record = Prediction(
        user_id=user_id,
        company_id=company.id,
        model_type=selected,
        horizon_days=horizon_days,
        current_price=current_price,
        predicted_price=final_price,
        predicted_change_pct=change_pct,
        result_json=json.dumps(result),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    result["id"] = record.id
    result["created_at"] = record.created_at.isoformat() if record.created_at else None
    return result
