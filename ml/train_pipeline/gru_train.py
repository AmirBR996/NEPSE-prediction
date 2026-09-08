import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent

if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

import torch
import torch.nn as nn
from torch.optim import Adam

from data_preprocessing.feature import build_pipeline
from model.gru import StockGRU


def main(data):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader, scaler_X, scaler_y = build_pipeline(
        csv_path=str(PROJECT_ROOT / "data" / f"{data}.csv"),
        batch_size=32,
    )

    X_sample, _ = next(iter(train_loader))
    input_dim = X_sample.shape[-1]

    print("Input features:", input_dim)

    model = StockGRU(
        input_size=input_dim,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = Adam(
        model.parameters(),
        lr=0.0001
    )

    epochs = 100

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

    model_dir = PROJECT_ROOT / "trained_models" / data
    model_dir.mkdir(parents=True, exist_ok=True)

    save_path = model_dir / "stock_gru.pth"

    torch.save(
        model.state_dict(),
        save_path
    )

    print(f"Model saved to: {save_path}")