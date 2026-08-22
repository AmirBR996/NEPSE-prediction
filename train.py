import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import joblib
import torch
import torch.nn as nn
from torch.optim import Adam

from feature import build_pipeline
from model import StockLSTM


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Using device:", device)

  # 1. Load pipeline and get fitted scalers
  train_loader, test_loader, scaler_X, scaler_y = build_pipeline(
      csv_path=str(ROOT / "data" / "ADBL.csv"),
      batch_size=32,
  )

  # 2. Dynamically extract feature dimension from first batch
  X_sample, _ = next(iter(train_loader))
  input_dim = X_sample.shape[-1]

  model = StockLSTM(
      input_size=input_dim,
      hidden_size=64,
      num_layers=2,
      dropout=0.2,
  ).to(device)

  criterion = nn.MSELoss()
  optimizer = Adam(model.parameters(), lr=0.0001)
  epochs = 300

  # 3. Training loop
  for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for X, y in train_loader:
      X = X.to(device)
      y = y.to(device)

      optimizer.zero_grad()
      output = model(X)
      loss = criterion(output, y)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
      for X, y in test_loader:
        X = X.to(device)
        y = y.to(device)
        output = model(X)
        loss = criterion(output, y)
        test_loss += loss.item()

    test_loss /= len(test_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.6f} "
        f"Test Loss: {test_loss:.6f}"
    )

  # 4. Save Model Checkpoint and Scalers
  save_path = ROOT / "stock_lstm.pth"
  torch.save(model.state_dict(), save_path)
  joblib.dump(scaler_X, ROOT / "scaler_X.joblib")
  joblib.dump(scaler_y, ROOT / "scaler_y.joblib")

  print(f"Model saved to: {save_path}")
  print("Scalers saved successfully.")


if __name__ == "__main__":
  main()