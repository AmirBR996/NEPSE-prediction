import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def features(df):
    df["published_date"] = pd.to_datetime(df["published_date"])
    df = df.sort_values("published_date").reset_index(drop=True)

    df["close_lag_1"] = df["close"].shift(1)
    df["close_lag_5"] = df["close"].shift(5)
    df["close_lag_10"] = df["close"].shift(10)
    df["target"] = df["close"].shift(-1)

    df = df.dropna(
        subset=["close_lag_1", "close_lag_5", "close_lag_10", "target"]
    ).reset_index(drop=True)

    return df


def splitting_and_processing(df):
    train = df[df["published_date"] < "2026-01-01"].copy()
    test = df[df["published_date"] >= "2026-01-01"].copy()

    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "traded_quantity",
        "traded_amount",
        "close_lag_1",
        "close_lag_5",
        "close_lag_10",
    ]

    X_train = train[feature_cols].values.astype(float)
    y_train = train["target"].values.reshape(-1, 1).astype(float)

    X_test = test[feature_cols].values.astype(float)
    y_test = test["target"].values.reshape(-1, 1).astype(float)

    return X_train, y_train, X_test, y_test


def standardization(X_train, X_test, y_train, y_test):
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    return X_train, X_test, y_train, y_test, X_scaler, y_scaler


def tensor_conversion(X_train, X_test, y_train, y_test):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_test, y_train, y_test


def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def build_pipeline(csv_path="../data/ACLBSL.csv", batch_size=32):
    df = pd.read_csv(csv_path)
    df = features(df)

    X_train, y_train, X_test, y_test = splitting_and_processing(df)
    X_train, X_test, y_train, y_test, X_scaler, y_scaler = standardization(
        X_train, X_test, y_train, y_test
    )
    X_train, X_test, y_train, y_test = tensor_conversion(
        X_train, X_test, y_train, y_test
    )
    train_loader, test_loader = create_dataloaders(
        X_train, X_test, y_train, y_test, batch_size=batch_size
    )
    return train_loader, test_loader, X_scaler, y_scaler


if __name__ == "__main__":
    train_loader, test_loader, _, _ = build_pipeline()

    print("Train loader batches:", len(train_loader))
    print("Test loader batches:", len(test_loader))

    X_batch, y_batch = next(iter(train_loader))
    print("X_batch:", X_batch.shape)
    print("y_batch:", y_batch.shape)