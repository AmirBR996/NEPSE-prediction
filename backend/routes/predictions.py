from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.deps import require_authorized_user, require_user
from backend.model import Prediction, User
from backend.services.companies import get_company_by_symbol
from backend.services.prediction import predict_next_days

router = APIRouter(prefix="/predictions", tags=["Predictions"])


class PredictionRequest(BaseModel):
    company: str
    model: str | None = None
    horizon_days: int = Field(default=1, ge=1, le=7)


@router.post("")
@router.post("/")
def create_prediction(
    payload: PredictionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_authorized_user),
):
    company = get_company_by_symbol(db, payload.company)
    if not company or not company.is_active:
        raise HTTPException(status_code=404, detail="Company not found")

    try:
        return predict_next_days(
            db=db,
            company=company,
            user_id=user.id,
            model_type=payload.model,
            horizon_days=payload.horizon_days,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("")
@router.get("/")
def list_predictions(
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    rows = (
        db.query(Prediction)
        .filter(Prediction.user_id == user.id)
        .order_by(Prediction.id.desc())
        .limit(50)
        .all()
    )
    return {
        "predictions": [
            {
                "id": p.id,
                "company": p.company.symbol if p.company else None,
                "model_type": p.model_type,
                "horizon_days": p.horizon_days,
                "current_price": p.current_price,
                "predicted_price": p.predicted_price,
                "predicted_change_pct": p.predicted_change_pct,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in rows
        ]
    }


@router.get("/{prediction_id}")
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_user),
):
    row = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id, Prediction.user_id == user.id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")

    import json

    payload = {}
    if row.result_json:
        try:
            payload = json.loads(row.result_json)
        except json.JSONDecodeError:
            payload = {}

    return {
        "id": row.id,
        "company": row.company.symbol if row.company else None,
        "model_type": row.model_type,
        "horizon_days": row.horizon_days,
        "current_price": row.current_price,
        "predicted_price": row.predicted_price,
        "predicted_change_pct": row.predicted_change_pct,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "result": payload,
    }
