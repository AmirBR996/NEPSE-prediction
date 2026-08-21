import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.optim import Adam

from model.Feature_engineering import build_pipeline
from model.lstm import StockLSTM


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, _, _ = build_pipeline(
        csv_path=str(Path(__file__).resolve().parents[1] / "data.csv"),
        batch_size=32,
    )

    model = StockLSTM(
        input_size=9,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
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

    save_path = Path(__file__).resolve().parent / "stock_lstm.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()