import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model.features import (
    build_dataframe,
    FEATURE_COLUMNS,
    SequenceDataset,
    build_sequence_indices,
)
from model.lstm_v2 import LSTMRegressor


def inverse_transform(scaler, values):
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    return scaler.inverse_transform(values).reshape(-1)


def prepare_test_data(csv_path, seq_len, train_cutoff, val_cutoff):
    df, le = build_dataframe(csv_path)
    train_row_mask = df["published_date"] < train_cutoff

    feature_frame = df[FEATURE_COLUMNS]
    target_frame = df[["target"]]

    scaler_X_path = ROOT / "scaler_X.joblib"
    scaler_y_path = ROOT / "scaler_y.joblib"
    if not scaler_X_path.exists() or not scaler_y_path.exists():
        raise FileNotFoundError("Scalers not found. Train the model first.")

    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    features_scaled = scaler_X.transform(feature_frame).astype(np.float32)
    targets_scaled = scaler_y.transform(target_frame).astype(np.float32).reshape(-1)

    _, _, test_idx = build_sequence_indices(df, seq_len, train_cutoff, val_cutoff)
    test_ds = SequenceDataset(features_scaled, targets_scaled, test_idx, seq_len)

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=2 if sys.platform == "linux" else 0,
        pin_memory=torch.cuda.is_available(),
    )

    return df, test_idx, test_loader, scaler_y


def load_model(model_path, device):
    model = LSTMRegressor(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def evaluate_model(csv_path, model_path, save_plot_path, save_json_path):
    SEQ_LEN = 30
    TRAIN_CUTOFF = pd.Timestamp("2023-01-01")
    VAL_CUTOFF = pd.Timestamp("2025-01-01")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, test_idx, test_loader, scaler_y = prepare_test_data(
        csv_path, SEQ_LEN, TRAIN_CUTOFF, VAL_CUTOFF
    )
    model = load_model(Path(model_path), device)

    actual_scaled = []
    predicted_scaled = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            pred_batch = model(x_batch)
            actual_scaled.append(y_batch.cpu().numpy())
            predicted_scaled.append(pred_batch.cpu().numpy())

    y_true = np.concatenate(actual_scaled, axis=0).reshape(-1, 1)
    y_pred = np.concatenate(predicted_scaled, axis=0).reshape(-1, 1)

    y_true_original = inverse_transform(scaler_y, y_true)
    y_pred_original = inverse_transform(scaler_y, y_pred)

    current_close = df.loc[test_idx, "close"].values

    mae = float(mean_absolute_error(y_true_original, y_pred_original))
    rmse = float(np.sqrt(mean_squared_error(y_true_original, y_pred_original)))
    mape = float(
        np.mean(np.abs((y_true_original - y_pred_original) / (np.abs(y_true_original) + 1e-8)))
        * 100
    )
    r2 = float(r2_score(y_true_original, y_pred_original))

    actual_change = y_true_original - current_close
    pred_change = y_pred_original - current_close
    directional_acc = float(np.mean(np.sign(actual_change) == np.sign(pred_change)) * 100)

    plot_dates = df.loc[test_idx, "published_date"].values
    plt.figure(figsize=(14, 6))
    plt.plot(plot_dates, y_true_original, label="Actual", color="blue", linewidth=2)
    plt.plot(plot_dates, y_pred_original, label="Predicted", color="orange", linewidth=2, linestyle="--")
    plt.title("Global LSTM Test Prediction")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=200)
    plt.close()

    results = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "directional_accuracy": directional_acc,
        "test_samples": int(len(y_true_original)),
        "actual": y_true_original.tolist(),
        "predicted": y_pred_original.tolist(),
        "dates": [str(d) for d in plot_dates],
    }

    with open(save_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("======= Global LSTM Evaluation =======")
    print(f"Test samples     : {len(y_true_original)}")
    print(f"MAE              : {mae:.4f}")
    print(f"RMSE             : {rmse:.4f}")
    print(f"MAPE             : {mape:.2f}%")
    print(f"R2 Score         : {r2:.4f}")
    print(f"Directional Acc  : {directional_acc:.2f}%")
    print(f"Plot saved to    : {save_plot_path}")
    print(f"JSON saved to    : {save_json_path}")

    return results


if __name__ == "__main__":
    csv_path = str(ROOT / "data.csv")
    model_path = str(ROOT / "Train" / "lstm_best.pth")
    plot_path = str(ROOT / "Evaluation" / "global_lstm_prediction.png")
    json_path = str(ROOT / "Evaluation" / "global_lstm_metrics.json")

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train the model first."
        )

    evaluate_model(csv_path, model_path, plot_path, json_path)
