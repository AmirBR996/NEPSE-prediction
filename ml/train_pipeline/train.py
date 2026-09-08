import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import torch
import torch.nn as nn
from torch.optim import Adam

from ml.data_preprocessing.feature import build_pipeline
from ml.model.lstm import StockLSTM 
from ml.model.gru import StockGRU
from ml.model.transformer import StockTransformer


def main(
    data: str,
    model_name: str,
    epochs: int = 100,
    lr: float = 1e-4,
    include_test: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, scaler_X, scaler_y = build_pipeline(
        csv_path=str(ROOT / "data" / f"{data}.csv"),
        batch_size=32,
        include_test=include_test,
    )

    X_sample, _ = next(iter(train_loader))
    input_dim = X_sample.shape[-1]
    print(f"Input feature dimension: {input_dim}")

    if model_name == "lstm":
        model = StockLSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
        )
    elif model_name == "gru":
        model = StockGRU(
            input_size=input_dim,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
        )
    elif model_name == "transformer":
        model = StockTransformer(
            input_size=input_dim,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.2,
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    model_dir = ROOT / "trained_models" / data
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"stock_{model_name}.pth"
    scaler_X_path = model_dir / "scaler_X.joblib"
    scaler_y_path = model_dir / "scaler_y.joblib"

    best_test_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(X)

            # Ensure shapes align to avoid implicit broadcasting issues
            if output.shape != y.shape:
                y = y.view_as(output)

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Evaluation Phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)

                if output.shape != y.shape:
                    y = y.view_as(output)

                loss = criterion(output, y)
                test_loss += loss.item()

        test_loss /= len(test_loader)

        print(
            f"Epoch [{epoch + 1:03d}/{epochs}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Test Loss: {test_loss:.6f}"
        )

    torch.save(model.state_dict(), model_path)

    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)

    print("\nTraining Complete.")
    print(f"Best Model saved to: {model_path} (Test Loss: {best_test_loss:.6f})")
    print(f"Scalers saved to: {model_dir}")
