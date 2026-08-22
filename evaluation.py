import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from feature import build_pipeline
from model import StockLSTM


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Using device:", device)

  csv_path = str(ROOT / "data" / "ADBL.csv")
  model_path = ROOT / "stock_lstm.pth"
  scaler_y_path = ROOT / "scaler_y.joblib"

  # 1. Load Data Pipeline
  train_loader, test_loader, scaler_X, scaler_y = build_pipeline(
      csv_path=csv_path, batch_size=32, seq_length=30
  )

  if scaler_y_path.exists():
    scaler_y = joblib.load(scaler_y_path)

  # 2. Extract DataFrame details for price reconstruction
  df = pd.read_csv(csv_path)
  df["published_date"] = pd.to_datetime(df["published_date"])
  df = df.sort_values("published_date").reset_index(drop=True)

  # Build target returns same as feature.py to match exact indices
  df["target"] = df["close"].pct_change().shift(-1)
  df = df.dropna(
      subset=["close", "target"]
  ).reset_index(drop=True)

  # Filter test dataset matching feature.py threshold
  test_df = df[df["published_date"] >= "2026-01-01"].copy()

  # 3. Model Setup
  X_sample, _ = next(iter(test_loader))
  input_dim = X_sample.shape[-1]

  model = StockLSTM(
      input_size=input_dim, hidden_size=64, num_layers=2, dropout=0.2
  ).to(device)

  model.load_state_dict(torch.load(model_path, map_location=device))
  model.eval()

  # 4. Generate Predictions
  all_preds = []
  all_targets = []

  with torch.no_grad():
    for X_batch, y_batch in test_loader:
      X_batch = X_batch.to(device)
      pred = model(X_batch)
      all_preds.append(pred.cpu().numpy())
      all_targets.append(y_batch.numpy())

  preds_scaled = np.concatenate(all_preds, axis=0)
  targets_scaled = np.concatenate(all_targets, axis=0)

  # Unscale predicted and actual returns
  pred_returns = scaler_y.inverse_transform(preds_scaled).flatten()
  actual_returns = scaler_y.inverse_transform(targets_scaled).flatten()

  # 5. Reconstruct Actual Stock Prices
  n_samples = len(pred_returns)
  
  # Get exact dates and preceding prices corresponding to sequence outputs
  eval_df = test_df.iloc[-n_samples:].copy()
  plot_dates = eval_df["published_date"].values
  actual_prices = eval_df["close"].values
  
  # Calculate actual previous day close prices: Prev_Close = Current_Close / (1 + Actual_Return)
  prev_close_prices = actual_prices / (1.0 + actual_returns)

  # Reconstruct Predicted Prices: Pred_Price = Prev_Close * (1 + Pred_Return)
  predicted_prices = prev_close_prices * (1.0 + pred_returns)

  # 6. Price Metrics
  mae_price = np.mean(np.abs(predicted_prices - actual_prices))
  rmse_price = np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))
  direction_acc = np.mean(np.sign(pred_returns) == np.sign(actual_returns)) * 100

  print("--- Price Reconstruction Evaluation ---")
  print(f"Price MAE:             ${mae_price:.2f}")
  print(f"Price RMSE:            ${rmse_price:.2f}")
  print(f"Directional Accuracy:  {direction_acc:.2f}%")

  # 7. Plot Stock Prices Over Time
  plt.figure(figsize=(12, 5))
  plt.plot(
      plot_dates, actual_prices, label="Actual Price", color="blue", linewidth=1.5
  )
  plt.plot(
      plot_dates,
      predicted_prices,
      label="Predicted Price",
      color="darkorange",
      linestyle="--",
      linewidth=1.5,
  )

  plt.title("ADBL Stock Price: Actual vs Predicted")
  plt.xlabel("Date")
  plt.ylabel("Stock Price")
  plt.legend()
  plt.grid(alpha=0.3)
  plt.xticks(rotation=45)
  plt.tight_layout()

  plt.savefig("time_series_forecasting.png", dpi=150)
  plt.close()
  print("Saved price forecast plot to time_series_forecasting.png")


if __name__ == "__main__":
  main()