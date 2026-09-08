from fastapi import FastAPI
from backend.routes.scrape import router as scrape_router
from backend.routes.train import router as train_router
from backend.routes.auth import router as auth_router
from backend.routes.admin import router as admin_router

app = FastAPI(
    title="Stock backend API",
    version="1.0.0",
    description="Backend for stock price prediction"
)

app.include_router(scrape_router)
app.include_router(auth_router)
app.include_router(train_router)
app.include_router(admin_router)


@app.get("/")
def root():
    return {"status": "ok", "message": "Hello this is  Stock prediction backend!"}