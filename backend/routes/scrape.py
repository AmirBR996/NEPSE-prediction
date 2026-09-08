from fastapi import APIRouter, HTTPException
from scraper.nepse_data_scraper import main

router = APIRouter(
    prefix="/scrape",
    tags=["scrape"]
)


@router.post("/")
def datascrape():
    try:
        main()
        return {"response": "Data scrape successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )