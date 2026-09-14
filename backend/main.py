from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.database import Base, SessionLocal, engine
from backend import model  # noqa: F401 - register SQLAlchemy models
from backend.routes.scrape import router as scrape_router
from backend.routes.train import router as train_router
from backend.routes.auth import router as auth_router
from backend.routes.admin import router as admin_router
from backend.routes.companies import router as companies_router
from backend.routes.predictions import router as predictions_router
from backend.routes.training_jobs import router as training_router
from backend.seed import bootstrap

app = FastAPI(
    title="StockPulse API",
    version="2.0.0",
    description="Stock forecasting platform with auth, admin, and ML pipelines",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Include API Routers
app.include_router(auth_router)
app.include_router(companies_router)
app.include_router(predictions_router)
app.include_router(training_router)
app.include_router(scrape_router)
app.include_router(train_router)
app.include_router(admin_router)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        bootstrap(db)
    finally:
        db.close()


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


# 2. Serve index.html specifically at the root route
@app.get("/", response_class=FileResponse)
def root():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "status": "ok",
        "message": "StockPulse forecasting API",
        "docs": "/docs",
    }


# 3. Mount FRONTEND_DIR to root "/" AFTER API routes so static assets (style.css, app.js, predict.html) resolve properly
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")