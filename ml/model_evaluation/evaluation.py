import sys
from pathlib import Path

ML_ROOT=Path(__file__).resolve().parents[1]
PROJECT_ROOT=ML_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ml.data_preprocessing.feature import build_pipeline
from ml.model.lstm import StockLSTM
from ml.model.gru import StockGRU
from ml.model.transformer import StockTransformer
from ml.model_paths import (
    PURPOSE_EVALUATION,
    TEST_SPLIT_DATE,
    resolve_model_path,
)

def evaluate_model(model,test_loader,scaler_y,device):
    model.eval()
    all_preds=[]
    all_targets=[]

    with torch.no_grad():
        for X_batch,y_batch in test_loader:
            X_batch=X_batch.to(device)
            pred=model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds_scaled=np.concatenate(all_preds,axis=0)
    targets_scaled=np.concatenate(all_targets,axis=0)

    pred_returns=scaler_y.inverse_transform(preds_scaled).flatten()
    actual_returns=scaler_y.inverse_transform(targets_scaled).flatten()

    return pred_returns,actual_returns

def reconstruct_prices(pred_returns,actual_returns,test_df):
    n_samples=len(pred_returns)
    eval_df=test_df.iloc[-n_samples:].copy()

    plot_dates=eval_df["published_date"].values
    actual_prices=eval_df["close"].values

    prev_close_prices=actual_prices/(1.0+actual_returns)

    predicted_prices=prev_close_prices*(1.0+pred_returns)

    return predicted_prices,actual_prices,plot_dates

def calculate_metrics(predicted_prices,actual_prices,pred_returns,actual_returns):
    mae_price=np.mean(np.abs(predicted_prices-actual_prices))

    rmse_price=np.sqrt(
        np.mean((predicted_prices-actual_prices)**2)
    )

    mape_denom=np.where(np.abs(actual_prices)<1e-8,1e-8,np.abs(actual_prices))
    mape_price=np.mean(np.abs((actual_prices-predicted_prices)/mape_denom))*100

    ss_res=np.sum((actual_prices-predicted_prices)**2)
    ss_tot=np.sum((actual_prices-np.mean(actual_prices))**2)
    r2=float(1.0-(ss_res/ss_tot)) if ss_tot>0 else 0.0

    direction_acc=np.mean(
        np.sign(pred_returns)==np.sign(actual_returns)
    )*100

    return {
        "mae":float(mae_price),
        "rmse":float(rmse_price),
        "mape":float(mape_price),
        "r2":r2,
        "directional_accuracy":float(direction_acc)
    }

def load_model(data,model_name,input_dim,device,purpose=PURPOSE_EVALUATION):
    if model_name=="lstm":
        model=StockLSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
    elif model_name=="gru":
        model=StockGRU(
            input_size=input_dim,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        )
    elif model_name=="transformer":
        model=StockTransformer(
            input_size=input_dim,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.2
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model_path=resolve_model_path(data, model_name, purpose)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model=model.to(device)
    model.eval()

    return model

def evaluate(data,model_name,include_series=False):
    device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    csv_path=PROJECT_ROOT/"data"/f"{data}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data not found: {csv_path}"
        )

    train_loader,test_loader,scaler_X,scaler_y=build_pipeline(
        csv_path=str(csv_path),
        batch_size=32,
        include_test=True
    )

    if test_loader is None:
        raise ValueError("Test loader is required for evaluation")

    X_sample,_=next(iter(train_loader))
    input_dim=X_sample.shape[-1]

    model=load_model(
        data,
        model_name,
        input_dim,
        device,
        purpose=PURPOSE_EVALUATION,
    )

    pred_returns,actual_returns=evaluate_model(
        model,
        test_loader,
        scaler_y,
        device
    )

    df=pd.read_csv(csv_path)
    df["published_date"]=pd.to_datetime(
        df["published_date"]
    )
    df=df.sort_values("published_date").reset_index(drop=True)

    test_df=df[
        df["published_date"]>=TEST_SPLIT_DATE
    ].copy()

    predicted_prices,actual_prices,plot_dates=reconstruct_prices(
        pred_returns,
        actual_returns,
        test_df
    )

    metrics=calculate_metrics(
        predicted_prices,
        actual_prices,
        pred_returns,
        actual_returns
    )

    result={
        "company":data,
        "model":model_name,
        "metrics":metrics,
        "purpose":PURPOSE_EVALUATION,
        "test_split_date":TEST_SPLIT_DATE,
    }

    if include_series:
        dates=[]
        for d in plot_dates:
            if hasattr(d,"strftime"):
                dates.append(d.strftime("%Y-%m-%d"))
            else:
                dates.append(str(pd.Timestamp(d).date()))

        result["series"]={
            "dates":dates,
            "actual":[float(x) for x in actual_prices],
            "predicted":[float(x) for x in predicted_prices],
        }

    return result