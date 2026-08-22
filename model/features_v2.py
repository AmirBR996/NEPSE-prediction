import numpy as np
import pandas as pd
import torch
from typing import Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from torch.utils.data import Dataset


FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "per_change",
    "traded_quantity",
    "traded_amount",
    "status",
    "close_lag_1",
    "close_lag_5",
    "close_lag_10",
    "close_lag_20",
    "return_1",
    "return_5",
    "return_10",
    "return_20",
    "sma_5",
    "sma_10",
    "sma_20",
    "ema_5",
    "ema_10",
    "ema_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "bb_pct",
    "atr_14",
    "vol_sma_5",
    "vol_ratio_5",
    "vol_ratio_10",
]


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for w in (5, 10, 20, 50):
        df[f"sma_{w}"] = df.groupby("company_name")["close"].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )

    for w in (5, 10, 20):
        df[f"ema_{w}"] = df.groupby("company_name")["close"].transform(
            lambda x: x.ewm(span=w, adjust=False).mean()
        )

    delta = df.groupby("company_name")["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(df["company_name"]).transform(
        lambda x: x.rolling(window=14, min_periods=14).mean()
    )
    avg_loss = loss.groupby(df["company_name"]).transform(
        lambda x: x.rolling(window=14, min_periods=14).mean()
    )
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = df.groupby("company_name")["close"].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    ema_26 = df.groupby("company_name")["close"].transform(
        lambda x: x.ewm(span=26, adjust=False).mean()
    )
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df.groupby("company_name")["macd"].transform(
        lambda x: x.ewm(span=9, adjust=False).mean()
    )
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    sma_20 = df.groupby("company_name")["close"].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    std_20 = df.groupby("company_name")["close"].transform(
        lambda x: x.rolling(window=20, min_periods=1).std()
    )
    df["bb_upper"] = sma_20 + 2 * std_20
    df["bb_lower"] = sma_20 - 2 * std_20
    bb_width_denom = sma_20.replace(0, 1e-9)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_width_denom
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, 1e-9)
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / bb_range

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df.groupby("company_name")["close"].shift(1)).abs()
    low_close = (df["low"] - df.groupby("company_name")["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.groupby(df["company_name"]).transform(
        lambda x: x.rolling(window=14, min_periods=14).mean()
    )

    df["vol_sma_5"] = df.groupby("company_name")["traded_quantity"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    df["vol_ratio_5"] = df["traded_quantity"] / df["vol_sma_5"].replace(0, 1e-9)
    df["vol_ratio_10"] = df["traded_quantity"] / df.groupby("company_name")[
        "traded_quantity"
    ].transform(lambda x: x.rolling(window=10, min_periods=1).mean()).replace(0, 1e-9)

    return df


def build_dataframe(csv_path: str) -> Tuple[pd.DataFrame, LabelEncoder]:
    df = pd.read_csv(csv_path)
    df["published_date"] = pd.to_datetime(df["published_date"])
    df = df.sort_values(["company_name", "published_date"]).reset_index(drop=True)

    df = df.dropna(subset=["per_change"]).reset_index(drop=True)

    le = LabelEncoder()
    df["company_id"] = le.fit_transform(df["company_name"].astype(str))

    g = df.groupby("company_name", sort=False)
    df["close_lag_1"] = g["close"].shift(1)
    df["close_lag_5"] = g["close"].shift(5)
    df["close_lag_10"] = g["close"].shift(10)
    df["close_lag_20"] = g["close"].shift(20)
    df["return_1"] = g["close"].pct_change(1)
    df["return_5"] = g["close"].pct_change(5)
    df["return_10"] = g["close"].pct_change(10)
    df["return_20"] = g["close"].pct_change(20)

    df["target"] = g["close"].shift(-1)
    df["target_date"] = g["published_date"].shift(-1)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = add_technical_indicators(df)

    df = df.dropna(
        subset=FEATURE_COLUMNS + ["target", "target_date", "company_id", "published_date"]
    ).reset_index(drop=True)

    return df, le


class SequenceDataset(Dataset):
    def __init__(self, features, targets, end_indices, sequence_length):
        self.features = np.ascontiguousarray(features, dtype=np.float32)
        self.targets = np.ascontiguousarray(targets, dtype=np.float32)
        self.end_indices = np.asarray(end_indices, dtype=np.int64)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.end_indices)

    def __getitem__(self, index):
        end = int(self.end_indices[index])
        start = end - self.sequence_length + 1
        x = torch.from_numpy(self.features[start : end + 1])
        y = torch.from_numpy(self.targets[end : end + 1])
        return x, y


def build_sequence_indices(df, sequence_length, train_cutoff, val_cutoff):
    train_indices = []
    val_indices = []
    test_indices = []

    for _, idx in df.groupby("company_id", sort=False).indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) < sequence_length:
            continue

        valid = idx[sequence_length - 1 :]
        dates = df.loc[valid, "published_date"].values

        train_mask = dates < train_cutoff
        val_mask = (dates >= train_cutoff) & (dates < val_cutoff)
        test_mask = dates >= val_cutoff

        train_indices.append(valid[train_mask])
        val_indices.append(valid[val_mask])
        test_indices.append(valid[test_mask])

    def safe_concat(arrays):
        return np.concatenate(arrays) if arrays else np.array([], dtype=np.int64)

    return safe_concat(train_indices), safe_concat(val_indices), safe_concat(test_indices)
