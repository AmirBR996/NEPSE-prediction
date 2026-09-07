import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.optim import Adam

from data_preprocessing.feature import build_pipeline
from model.transformer import StockTransformer


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    train_loader, test_loader, scaler_X, scaler_y = (
        build_pipeline(
            csv_path=str(
                ROOT / "data" / "ADBL.csv"
            ),
            batch_size=32,
            seq_length=30,
        )
    )

    X_sample, y_sample = next(
        iter(train_loader)
    )

    input_dim = X_sample.shape[-1]

    print(
        "Input shape:",
        X_sample.shape
    )

    print(
        "Target shape:",
        y_sample.shape
    )

    print(
        "Input features:",
        input_dim
    )

    model = StockTransformer(
        input_size=input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2,
    ).to(device)

    print("\nModel:")
    print(model)

    criterion = nn.MSELoss()

    optimizer = Adam(
        model.parameters(),
        lr=0.0001
    )

    epochs = 300

    for epoch in range(epochs):

        model.train()

        train_loss = 0.0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(X)

            loss = criterion(
                output,
                y
            )

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

                loss = criterion(
                    output,
                    y
                )

                test_loss += loss.item()

        test_loss /= len(test_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Test Loss: {test_loss:.6f}"
        )

    model_dir = (
        ROOT / "trained_models"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_path = (
        model_dir
        / "stock_transformer.pth"
    )

    torch.save(
        model.state_dict(),
        save_path
    )

    print(
        f"\nTransformer model saved to:"
        f"\n{save_path}"
    )


if __name__ == "__main__":
    main()