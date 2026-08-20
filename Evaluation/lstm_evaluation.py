import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.Feature_engineering import features, splitting_and_processing, standardization
from model.lstm import StockLSTM


def prepare_test_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df = features(df)

    X_train, y_train, X_test, y_test = splitting_and_processing(df)
    _, X_test_scaled, _, y_test_scaled, _, y_scaler = standardization(
        X_train, X_test, y_train, y_test
    )

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    test_dates = df[df["published_date"] >= "2026-01-01"]["published_date"].reset_index(drop=True)
    return test_loader, y_scaler, test_dates


def load_model(model_path: Path, device: torch.device):
    model = StockLSTM(input_size=9, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def evaluate_model(csv_path: str, model_path: str, save_plot_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader, y_scaler, test_dates = prepare_test_data(csv_path)
    model = load_model(Path(model_path), device)

    actual_scaled = []
    predicted_scaled = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred_batch = model(X_batch)

            actual_scaled.append(y_batch.cpu().numpy())
            predicted_scaled.append(pred_batch.cpu().numpy())

    y_true = np.concatenate(actual_scaled, axis=0).reshape(-1, 1)
    y_pred = np.concatenate(predicted_scaled, axis=0).reshape(-1, 1)

    y_true_original = y_scaler.inverse_transform(y_true)
    y_pred_original = y_scaler.inverse_transform(y_pred)

    mae = mean_absolute_error(y_true_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
    mape = np.mean(np.abs((y_true_original - y_pred_original) / (np.abs(y_true_original) + 1e-8))) * 100
    r2 = r2_score(y_true_original, y_pred_original)

    plot_dates = list(test_dates.iloc[: len(y_true_original)])
    plt.figure(figsize=(14, 6))
    plt.plot(plot_dates, y_true_original, label="Actual", color="blue", linewidth=2)
    plt.plot(plot_dates, y_pred_original, label="Predicted", color="orange", linewidth=2, linestyle="--")
    plt.title("ACLBSL LSTM Test Prediction")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=200)
    plt.close()

    print("======= LSTM Evaluation =======")
    print(f"Test samples: {len(y_true_original)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2 Score: {r2:.4f}")
    print(f"Plot saved to: {save_plot_path}")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
        "actual": y_true_original.reshape(-1),
        "predicted": y_pred_original.reshape(-1),
    }


if __name__ == "__main__":
    csv_path = str(ROOT / "data" / "ACLBSL.csv")
    model_path = str(ROOT / "Train" / "stock_lstm.pth")
    plot_path = str(ROOT / "Evaluation" / "aclbsl_lstm_prediction.png")

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Please train the model first with the ACLBSL dataset."
        )

    evaluate_model(csv_path, model_path, plot_path)
