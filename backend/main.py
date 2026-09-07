from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import sys
import json
import uuid
import subprocess
import threading
import joblib
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.models.lstm import StockLSTM
from ml.models.gru import StockGRU
from ml.models.transformer import StockTransformer
from ml.data_preprocessing.feature import build_pipeline, features

app = FastAPI(title="NEPSE Neural Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "trained_models"
USERS_FILE = ROOT / "users.json"
TASKS = {}

if USERS_FILE.exists():
    with open(USERS_FILE) as f:
        USERS = json.load(f)
else:
    USERS = {}


class LoginReq(BaseModel):
    email: str
    password: str


class RegisterReq(BaseModel):
    name: str
    email: str
    password: str
    role: str = "user"


class PredictReq(BaseModel):
    symbol: str = "ADBL"
    model: str = "lstm"
    days: int = 7


@app.post("/api/auth/register")
def register(req: RegisterReq):
    if req.email in USERS:
        raise HTTPException(status_code=400, detail="User already exists")
    USERS[req.email] = {
        "name": req.name,
        "email": req.email,
        "password": req.password,
        "role": req.role,
    }
    with open(USERS_FILE, "w") as f:
        json.dump(USERS, f)
    return {"message": "Registered successfully"}


@app.post("/api/auth/login")
def login(req: LoginReq):
    user = USERS.get(req.email)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "email": user["email"],
        "name": user["name"],
        "role": user["role"],
        "token": str(uuid.uuid4()),
    }


@app.get("/api/dashboard/symbols")
def get_symbols():
    return {"symbols": sorted([f.stem for f in DATA_DIR.glob("*.csv")])}


@app.get("/api/dashboard/data")
def get_dashboard_data(symbol: str = "ADBL"):
    csv_path = DATA_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Data file not found")
    df = pd.read_csv(csv_path)
    df["published_date"] = pd.to_datetime(df["published_date"])
    df = df.sort_values("published_date").tail(200)
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    change = float(latest["close"]) - float(prev["close"])
    change_pct = (change / float(prev["close"])) * 100 if float(prev["close"]) != 0 else 0
    return {
        "dates": df["published_date"].dt.strftime("%Y-%m-%d").tolist(),
        "open": df["open"].tolist(),
        "high": df["high"].tolist(),
        "low": df["low"].tolist(),
        "close": df["close"].tolist(),
        "volume": df["traded_quantity"].tolist(),
        "latest": {
            "date": latest["published_date"].strftime("%Y-%m-%d"),
            "close": float(latest["close"]),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": int(latest["traded_quantity"]),
        },
    }


@app.get("/api/admin/models")
def list_models():
    models = []
    for name, fname in [
        ("LSTM", "stock_lstm.pth"),
        ("GRU", "stock_gru.pth"),
        ("Transformer", "stock_transformer.pth"),
    ]:
        if (MODELS_DIR / fname).exists():
            models.append(name)
    return {"models": models}


@app.post("/api/admin/scrape")
def trigger_scrape():
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "type": "scrape",
        "status": "running",
        "started": datetime.now().isoformat(),
    }

    def run():
        try:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scraper" / "nepse_data_scraper.py")],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(ROOT / "scraper"),
            )
            TASKS[task_id]["status"] = "completed" if result.returncode == 0 else "failed"
            TASKS[task_id]["output"] = result.stdout
            TASKS[task_id]["error"] = result.stderr
        except Exception as e:
            TASKS[task_id] = {
                "type": "scrape",
                "status": "failed",
                "error": str(e),
            }

    threading.Thread(target=run, daemon=True).start()
    return {"task_id": task_id, "status": "started"}


@app.post("/api/admin/train")
def trigger_training(model: str = "lstm"):
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {
        "type": "train",
        "model": model,
        "status": "running",
        "started": datetime.now().isoformat(),
    }
    scripts = {
        "lstm": "train/lstm_train.py",
        "gru": "train/gru_train.py",
        "transformer": "train/transformer_train.py",
    }
    if model not in scripts:
        raise HTTPException(status_code=400, detail="Invalid model")

    def run():
        try:
            result = subprocess.run(
                [sys.executable, str(ROOT / scripts[model])],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(ROOT),
            )
            TASKS[task_id]["status"] = "completed" if result.returncode == 0 else "failed"
            TASKS[task_id]["output"] = result.stdout
            TASKS[task_id]["error"] = result.stderr
        except Exception as e:
            TASKS[task_id] = {
                "type": "train",
                "model": model,
                "status": "failed",
                "error": str(e),
            }

    threading.Thread(target=run, daemon=True).start()
    return {"task_id": task_id, "status": "started"}


@app.get("/api/admin/tasks/{task_id}")
def get_task(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    return TASKS[task_id]


MODEL_CLASSES = {
    "lstm": StockLSTM,
    "gru": StockGRU,
    "transformer": StockTransformer,
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_scalers(symbol):
    for prefix in [f"{symbol}_scaler", "scaler"]:
        for base in [MODELS_DIR, ROOT]:
            sx = base / f"{prefix}_X.joblib"
            sy = base / f"{prefix}_y.joblib"
            if sx.exists() and sy.exists():
                return joblib.load(sx), joblib.load(sy)
    return None, None


def ensure_scalers(symbol, seq_length=30):
    scaler_X, scaler_y = load_scalers(symbol)
    if scaler_X is None:
        csv_path = DATA_DIR / f"{symbol}.csv"
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail=f"Data for {symbol} not found")
        _, _, scaler_X, scaler_y = build_pipeline(
            str(csv_path), batch_size=32, seq_length=seq_length
        )
    return scaler_X, scaler_y


@app.post("/api/prediction/predict")
def predict(req: PredictReq):
    device = get_device()
    model_cls = MODEL_CLASSES.get(req.model)
    if not model_cls:
        raise HTTPException(status_code=400, detail="Invalid model")

    model_path = MODELS_DIR / f"stock_{req.model}.pth"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {req.model} not trained yet")

    csv_path = DATA_DIR / f"{req.symbol}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Data for {req.symbol} not found")

    scaler_X, scaler_y = ensure_scalers(req.symbol)

    df = pd.read_csv(csv_path)
    df = features(df)

    if len(df) < 30:
        raise HTTPException(status_code=400, detail="Not enough data for prediction")

    feature_cols = [
        "open",
        "high",
        "low",
        "traded_quantity",
        "traded_amount",
        "return_1",
        "return_5",
        "return_10",
    ]
    last_seq = df[feature_cols].tail(30).values
    last_scaled = scaler_X.transform(last_seq)
    X = torch.tensor(last_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    model = model_cls(input_size=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        pred_scaled = model(X).cpu().numpy()

    pred_return = scaler_y.inverse_transform(pred_scaled)[0][0]
    last_close = float(df["close"].iloc[-1])
    predicted_price = last_close * (1 + pred_return)

    return {
        "symbol": req.symbol,
        "model": req.model,
        "last_close": last_close,
        "predicted_return": float(pred_return),
        "predicted_price": predicted_price,
        "last_date": df["published_date"].iloc[-1].strftime("%Y-%m-%d"),
    }


@app.get("/api/prediction/chart")
def prediction_chart(symbol: str = "ADBL", model: str = "lstm"):
    device = get_device()
    model_cls = MODEL_CLASSES.get(model)
    if not model_cls:
        raise HTTPException(status_code=400, detail="Invalid model")

    model_path = MODELS_DIR / f"stock_{model}.pth"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    csv_path = DATA_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Data not found")

    scaler_X, scaler_y = ensure_scalers(symbol)

    df = pd.read_csv(csv_path)
    df = features(df)
    df["published_date"] = pd.to_datetime(df["published_date"])

    feature_cols = [
        "open",
        "high",
        "low",
        "traded_quantity",
        "traded_amount",
        "return_1",
        "return_5",
        "return_10",
    ]

    test_df = df[df["published_date"] >= "2026-01-01"].copy()
    if len(test_df) < 35:
        raise HTTPException(status_code=400, detail="Not enough test data for chart")

    test_raw = test_df[feature_cols].values
    test_scaled = scaler_X.transform(test_raw)

    sequences = []
    for i in range(len(test_scaled) - 30):
        sequences.append(test_scaled[i : i + 30])

    X_test = torch.tensor(np.array(sequences), dtype=torch.float32).to(device)

    mdl = model_cls(input_size=len(feature_cols)).to(device)
    mdl.load_state_dict(torch.load(model_path, map_location=device))
    mdl.eval()

    with torch.no_grad():
        preds_scaled = mdl(X_test).cpu().numpy()

    pred_returns = scaler_y.inverse_transform(preds_scaled).flatten()
    actual_prices = test_df["close"].iloc[-len(pred_returns) :].values
    actual_returns = test_df["target"].iloc[-len(pred_returns) :].values
    prev_close = actual_prices / (1.0 + actual_returns)
    predicted_prices = prev_close * (1.0 + pred_returns)
    dates = test_df["published_date"].iloc[-len(pred_returns) :].dt.strftime("%Y-%m-%d").tolist()

    return {
        "dates": dates,
        "actual": actual_prices.tolist(),
        "predicted": predicted_prices.tolist(),
        "returns_actual": actual_returns.tolist(),
        "returns_predicted": pred_returns.tolist(),
    }


@app.get("/api/prediction/plot")
def get_comparison_plot():
    plot_path = ROOT / "lstm_gru_transformer_comparison.png"
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found. Run evaluation first.")
    return FileResponse(plot_path, media_type="image/png")


frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
