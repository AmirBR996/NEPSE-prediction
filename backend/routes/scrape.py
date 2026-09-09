from fastapi import APIRouter, Depends, HTTPException

from backend.deps import require_admin
from backend.model import User
from scraper.nepse_data_scraper import main

router = APIRouter(prefix="/scrape", tags=["scrape"])


@router.post("/")
def datascrape(_admin: User = Depends(require_admin)):
    try:
        main()
        return {"response": "Data scrape successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
