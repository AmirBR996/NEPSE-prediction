from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.services.companies import (
    company_history,
    company_overview,
    get_company_by_symbol,
    latest_evaluations,
    list_companies,
)

router = APIRouter(prefix="/companies", tags=["Companies"])


@router.get("")
@router.get("/")
def get_companies(db: Session = Depends(get_db)):
    return {"companies": list_companies(db)}


@router.get("/{company_id}")
def get_company(company_id: str, db: Session = Depends(get_db)):
    company = _resolve_company(db, company_id)
    return company_overview(db, company)


@router.get("/{company_id}/data")
def get_company_data(
    company_id: str,
    range: str = Query("1Y", alias="range"),
    db: Session = Depends(get_db),
):
    company = _resolve_company(db, company_id)
    try:
        return company_history(company.symbol, range)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/{company_id}/forecast")
def get_company_forecast(company_id: str, db: Session = Depends(get_db)):
    company = _resolve_company(db, company_id)
    overview = company_overview(db, company)
    evaluations = latest_evaluations(db, company)
    return {
        **overview,
        "forecasts": evaluations,
        "production_model": company.production_model,
    }


def _resolve_company(db: Session, company_id: str):
    company = None
    if company_id.isdigit():
        from backend.model import Company

        company = db.query(Company).filter(Company.id == int(company_id)).first()
    if not company:
        company = get_company_by_symbol(db, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return company
