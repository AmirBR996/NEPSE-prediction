import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def features(df):
  df["published_date"] = pd.to_datetime(df["published_date"])
  df = df.sort_values("published_date").reset_index(drop=True)

  # Target: Next-day percentage return
  df["target"] = df["close"].pct_change().shift(-1)

  # Stationarity / Returns Features
  df["return_1"] = df["close"].pct_change(1)
  df["return_5"] = df["close"].pct_change(5)
  df["return_10"] = df["close"].pct_change(10)

  # Clean NaNs caused by percentage shifts
  df = df.dropna(
      subset=["return_1", "return_5", "return_10", "target"]
  ).reset_index(drop=True)

  return df


def create_sequences(X_data, y_data, seq_length=30):
  """Converts 2D feature array into 3D sequence window tensors.

  Shape transformation:
    X: [N, num_features] -> [N - seq_length, seq_length, num_features]
    y: [N, 1]            -> [N - seq_length, 1]
  """
  X_seq, y_seq = [], []
  for i in range(len(X_data) - seq_length):
    X_seq.append(X_data[i : i + seq_length])
    y_seq.append(y_data[i + seq_length])

  return np.array(X_seq), np.array(y_seq)


def splitting_and_processing(df, seq_length=30):
  train = df[df["published_date"] < "2026-01-01"].copy()
  test = df[df["published_date"] >= "2026-01-01"].copy()

  feature_cols = [
      "open",
      "high",
      "low",
      "traded_quantity",
      "traded_amount",
      "return_1",
      "return_5",
      "return_10",
  ]

  X_train_raw = train[feature_cols].values
  y_train_raw = train["target"].values.reshape(-1, 1)

  X_test_raw = test[feature_cols].values
  y_test_raw = test["target"].values.reshape(-1, 1)

  # Scale features before windowing
  X_scaler = StandardScaler()
  y_scaler = StandardScaler()

  X_train_scaled = X_scaler.fit_transform(X_train_raw)
  X_test_scaled = X_scaler.transform(X_test_raw)

  y_train_scaled = y_scaler.fit_transform(y_train_raw)
  y_test_scaled = y_scaler.transform(y_test_raw)

  # Build 3D sliding sequence windows
  X_train_seq, y_train_seq = create_sequences(
      X_train_scaled, y_train_scaled, seq_length=seq_length
  )
  X_test_seq, y_test_seq = create_sequences(
      X_test_scaled, y_test_scaled, seq_length=seq_length
  )

  return (
      X_train_seq,
      y_train_seq,
      X_test_seq,
      y_test_seq,
      X_scaler,
      y_scaler,
  )


def tensor_conversion(X_train, y_train, X_test, y_test):
  X_train = torch.tensor(X_train, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.float32)

  X_test = torch.tensor(X_test, dtype=torch.float32)
  y_test = torch.tensor(y_test, dtype=torch.float32)

  return X_train, y_train, X_test, y_test


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
  train_dataset = TensorDataset(X_train, y_train)
  test_dataset = TensorDataset(X_test, y_test)

  # Keep shuffle=False for time-series memory stability
  train_loader = DataLoader(
      train_dataset, batch_size=batch_size, shuffle=False
  )
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, test_loader


def build_pipeline(csv_path, batch_size=32, seq_length=30):
  df = pd.read_csv(csv_path)
  df = features(df)

  (
      X_train_seq,
      y_train_seq,
      X_test_seq,
      y_test_seq,
      X_scaler,
      y_scaler,
  ) = splitting_and_processing(df, seq_length=seq_length)

  X_train, y_train, X_test, y_test = tensor_conversion(
      X_train_seq, y_train_seq, X_test_seq, y_test_seq
  )

  train_loader, test_loader = create_dataloaders(
      X_train, y_train, X_test, y_test, batch_size=batch_size
  )

  return train_loader, test_loader, X_scaler, y_scaler


if __name__ == "__main__":
  train_loader, test_loader, _, _ = build_pipeline(
      "../data/ADBL.csv", batch_size=32, seq_length=30
  )
  X_batch, y_batch = next(iter(train_loader))

  print("3D Sequence Batch Shape (X):", X_batch.shape)  # [32, 30, 8]
  print("Target Batch Shape (y):    ", y_batch.shape)  # [32, 1]