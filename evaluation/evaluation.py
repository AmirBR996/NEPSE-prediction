import sys
from pathlib import Path

# Project root: /home/amir/stock
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_preprocessing.feature import build_pipeline
from models.lstm import StockLSTM
from models.gru import StockGRU
from models.transformer import StockTransformer


def evaluate_model(model, test_loader, scaler_y, device):

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)

            pred = model(X_batch)

            all_preds.append(
                pred.cpu().numpy()
            )

            all_targets.append(
                y_batch.numpy()
            )

    preds_scaled = np.concatenate(
        all_preds,
        axis=0
    )

    targets_scaled = np.concatenate(
        all_targets,
        axis=0
    )

    # Convert scaled returns back to original returns
    pred_returns = scaler_y.inverse_transform(
        preds_scaled
    ).flatten()

    actual_returns = scaler_y.inverse_transform(
        targets_scaled
    ).flatten()

    return pred_returns, actual_returns


def reconstruct_prices(
    pred_returns,
    actual_returns,
    test_df
):

    n_samples = len(pred_returns)

    # Match samples with corresponding test rows
    eval_df = test_df.iloc[-n_samples:].copy()

    plot_dates = eval_df[
        "published_date"
    ].values

    actual_prices = eval_df[
        "close"
    ].values

    # Reconstruct previous day's close
    prev_close_prices = (
        actual_prices
        / (1.0 + actual_returns)
    )

    # Reconstruct predicted prices
    predicted_prices = (
        prev_close_prices
        * (1.0 + pred_returns)
    )

    return (
        predicted_prices,
        actual_prices,
        plot_dates
    )


def calculate_metrics(
    predicted_prices,
    actual_prices,
    pred_returns,
    actual_returns
):

    # Price MAE
    mae_price = np.mean(
        np.abs(
            predicted_prices
            - actual_prices
        )
    )

    # Price RMSE
    rmse_price = np.sqrt(
        np.mean(
            (
                predicted_prices
                - actual_prices
            ) ** 2
        )
    )

    # Directional Accuracy
    direction_acc = (
        np.mean(
            np.sign(pred_returns)
            == np.sign(actual_returns)
        )
        * 100
    )

    return (
        mae_price,
        rmse_price,
        direction_acc
    )


def main():

    # ==================================================
    # DEVICE
    # ==================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ==================================================
    # PATHS
    # ==================================================

    csv_path = (
        ROOT
        / "data"
        / "ADBL.csv"
    )

    lstm_model_path = (
        ROOT
        / "trained_models"
        / "stock_lstm.pth"
    )

    gru_model_path = (
        ROOT
        / "trained_models"
        / "stock_gru.pth"
    )

    transformer_model_path = (
        ROOT
        / "trained_models"
        / "stock_transformer.pth"
    )

    scaler_y_path = (
        ROOT
        / "scaler_y.joblib"
    )

    # ==================================================
    # LOAD DATA PIPELINE
    # ==================================================

    print("\nLoading data...")

    (
        train_loader,
        test_loader,
        scaler_X,
        scaler_y
    ) = build_pipeline(
        csv_path=str(csv_path),
        batch_size=32,
        seq_length=30
    )

    # Load same target scaler used during training
    if scaler_y_path.exists():

        scaler_y = joblib.load(
            scaler_y_path
        )

        print(
            "Loaded scaler_y.joblib"
        )

    else:

        print(
            "Warning: scaler_y.joblib not found."
        )

    # ==================================================
    # LOAD DATAFRAME
    # ==================================================

    df = pd.read_csv(
        csv_path
    )

    df["published_date"] = pd.to_datetime(
        df["published_date"]
    )

    df = (
        df
        .sort_values("published_date")
        .reset_index(drop=True)
    )

    # Same target calculation used in feature.py
    df["target"] = (
        df["close"]
        .pct_change()
        .shift(-1)
    )

    df = (
        df
        .dropna(
            subset=[
                "close",
                "target"
            ]
        )
        .reset_index(drop=True)
    )

    # ==================================================
    # TEST DATA
    # ==================================================

    test_df = df[
        df["published_date"]
        >= "2026-01-01"
    ].copy()

    print(
        "Test dataframe samples:",
        len(test_df)
    )

    # ==================================================
    # INPUT DIMENSION
    # ==================================================

    X_sample, _ = next(
        iter(test_loader)
    )

    input_dim = X_sample.shape[-1]

    print(
        "Input features:",
        input_dim
    )

    # ==================================================
    # LOAD LSTM
    # ==================================================

    print("\nLoading LSTM...")

    lstm = StockLSTM(
        input_size=input_dim,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)

    lstm.load_state_dict(
        torch.load(
            lstm_model_path,
            map_location=device
        )
    )

    lstm.eval()

    print(
        "LSTM loaded:",
        lstm_model_path
    )

    # ==================================================
    # LOAD GRU
    # ==================================================

    print("\nLoading GRU...")

    gru = StockGRU(
        input_size=input_dim,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)

    gru.load_state_dict(
        torch.load(
            gru_model_path,
            map_location=device
        )
    )

    gru.eval()

    print(
        "GRU loaded:",
        gru_model_path
    )

    # ==================================================
    # LOAD TRANSFORMER
    # ==================================================

    print("\nLoading Transformer...")

    transformer = StockTransformer(
        input_size=input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2
    ).to(device)

    transformer.load_state_dict(
        torch.load(
            transformer_model_path,
            map_location=device
        )
    )

    transformer.eval()

    print(
        "Transformer loaded:",
        transformer_model_path
    )

    # ==================================================
    # LSTM PREDICTIONS
    # ==================================================

    print(
        "\nGenerating LSTM predictions..."
    )

    (
        lstm_pred_returns,
        lstm_actual_returns
    ) = evaluate_model(
        lstm,
        test_loader,
        scaler_y,
        device
    )

    # ==================================================
    # GRU PREDICTIONS
    # ==================================================

    print(
        "Generating GRU predictions..."
    )

    (
        gru_pred_returns,
        gru_actual_returns
    ) = evaluate_model(
        gru,
        test_loader,
        scaler_y,
        device
    )

    # ==================================================
    # TRANSFORMER PREDICTIONS
    # ==================================================

    print(
        "Generating Transformer predictions..."
    )

    (
        transformer_pred_returns,
        transformer_actual_returns
    ) = evaluate_model(
        transformer,
        test_loader,
        scaler_y,
        device
    )

    # ==================================================
    # RECONSTRUCT LSTM PRICES
    # ==================================================

    (
        lstm_pred_prices,
        actual_prices,
        plot_dates
    ) = reconstruct_prices(
        lstm_pred_returns,
        lstm_actual_returns,
        test_df
    )

    # ==================================================
    # RECONSTRUCT GRU PRICES
    # ==================================================

    (
        gru_pred_prices,
        _,
        _
    ) = reconstruct_prices(
        gru_pred_returns,
        gru_actual_returns,
        test_df
    )

    # ==================================================
    # RECONSTRUCT TRANSFORMER PRICES
    # ==================================================

    (
        transformer_pred_prices,
        _,
        _
    ) = reconstruct_prices(
        transformer_pred_returns,
        transformer_actual_returns,
        test_df
    )

    # ==================================================
    # LSTM METRICS
    # ==================================================

    (
        lstm_mae,
        lstm_rmse,
        lstm_direction
    ) = calculate_metrics(
        lstm_pred_prices,
        actual_prices,
        lstm_pred_returns,
        lstm_actual_returns
    )

    # ==================================================
    # GRU METRICS
    # ==================================================

    (
        gru_mae,
        gru_rmse,
        gru_direction
    ) = calculate_metrics(
        gru_pred_prices,
        actual_prices,
        gru_pred_returns,
        gru_actual_returns
    )

    # ==================================================
    # TRANSFORMER METRICS
    # ==================================================

    (
        transformer_mae,
        transformer_rmse,
        transformer_direction
    ) = calculate_metrics(
        transformer_pred_prices,
        actual_prices,
        transformer_pred_returns,
        transformer_actual_returns
    )

    # ==================================================
    # PRINT RESULTS
    # ==================================================

    print("\n")
    print("=" * 60)
    print("              MODEL COMPARISON")
    print("=" * 60)

    print("\nLSTM")
    print("-" * 60)

    print(
        f"Price MAE:             "
        f"{lstm_mae:.4f}"
    )

    print(
        f"Price RMSE:            "
        f"{lstm_rmse:.4f}"
    )

    print(
        f"Directional Accuracy:  "
        f"{lstm_direction:.2f}%"
    )

    print("\nGRU")
    print("-" * 60)

    print(
        f"Price MAE:             "
        f"{gru_mae:.4f}"
    )

    print(
        f"Price RMSE:            "
        f"{gru_rmse:.4f}"
    )

    print(
        f"Directional Accuracy:  "
        f"{gru_direction:.2f}%"
    )

    print("\nTRANSFORMER")
    print("-" * 60)

    print(
        f"Price MAE:             "
        f"{transformer_mae:.4f}"
    )

    print(
        f"Price RMSE:            "
        f"{transformer_rmse:.4f}"
    )

    print(
        f"Directional Accuracy:  "
        f"{transformer_direction:.2f}%"
    )

    # ==================================================
    # WINNER BY METRIC
    # ==================================================

    print("\n")
    print("=" * 60)
    print("                 WINNERS")
    print("=" * 60)

    # MAE
    mae_results = {
        "LSTM": lstm_mae,
        "GRU": gru_mae,
        "Transformer": transformer_mae
    }

    best_mae_model = min(
        mae_results,
        key=mae_results.get
    )

    print(
        f"Lowest MAE: "
        f"{best_mae_model} "
        f"({mae_results[best_mae_model]:.4f})"
    )

    # RMSE
    rmse_results = {
        "LSTM": lstm_rmse,
        "GRU": gru_rmse,
        "Transformer": transformer_rmse
    }

    best_rmse_model = min(
        rmse_results,
        key=rmse_results.get
    )

    print(
        f"Lowest RMSE: "
        f"{best_rmse_model} "
        f"({rmse_results[best_rmse_model]:.4f})"
    )

    # Directional Accuracy
    direction_results = {
        "LSTM": lstm_direction,
        "GRU": gru_direction,
        "Transformer": transformer_direction
    }

    best_direction_model = max(
        direction_results,
        key=direction_results.get
    )

    print(
        f"Best Directional Accuracy: "
        f"{best_direction_model} "
        f"({direction_results[best_direction_model]:.2f}%)"
    )

    print("=" * 60)

    # ==================================================
    # SINGLE COMPARISON GRAPH
    # ==================================================

    plt.figure(
        figsize=(15, 7)
    )

    # Actual price
    plt.plot(
        plot_dates,
        actual_prices,
        label="Actual Price",
        linewidth=2.0
    )

    # LSTM
    plt.plot(
        plot_dates,
        lstm_pred_prices,
        label="LSTM Prediction",
        linestyle="--",
        linewidth=1.5
    )

    # GRU
    plt.plot(
        plot_dates,
        gru_pred_prices,
        label="GRU Prediction",
        linestyle=":",
        linewidth=1.8
    )

    # Transformer
    plt.plot(
        plot_dates,
        transformer_pred_prices,
        label="Transformer Prediction",
        linestyle="-.",
        linewidth=1.5
    )

    plt.title(
        "ADBL Stock Price: Actual vs LSTM vs GRU vs Transformer"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Stock Price"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    output_path = (
        ROOT
        / "lstm_gru_transformer_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    print(
        "\nComparison graph saved to:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()