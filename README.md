# Stock Price Prediction with Deep Learning

Predict next-day stock prices using sequence-based deep learning models (LSTM, GRU, Transformer) on NEPSE historical data.

## Project Structure

```
stock/
├── data/
│   └── ADBL.csv                 # ADBL historical OHLCV data (~4k rows)
├── data_preprocessing/
│   └── feature.py               # Feature engineering, sequences, scalers, DataLoaders
├── models/
│   ├── lstm.py                  # LSTM model
│   ├── gru.py                   # GRU model
│   └── transformer.py           # Transformer model with positional encoding
├── train/
│   ├── lstm_train.py            # Train LSTM
│   ├── gru_train.py             # Train GRU
│   └── transformer_train.py     # Train Transformer
├── evaluation/
│   └── evaluation.py            # Evaluate all 3 models and compare
├── trained_models/
│   ├── stock_lstm.pth
│   ├── stock_gru.pth
│   └── stock_transformer.pth
├── files/
│   ├── scaler_X.joblib
│   └── scaler_y.joblib
├── lstm_gru_transformer_comparison.png  # Comparison plot
├── requirement.txt
└── .gitignore
```

## Features

- **Sequence windowing**: 30-day sliding windows as input sequences
- **Return-based target**: Predicts next-day percentage return instead of raw price
- **Standardized features**: `StandardScaler` fitted on training data only
- **3 model architectures**:
  - LSTM (2-layer, hidden=64)
  - GRU (2-layer, hidden=64)
  - Transformer (2-layer, d_model=64, 4 heads)
- **Unified evaluation**: Price MAE, RMSE, and directional accuracy

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

## Usage

### Train a model

```bash
# LSTM
python train/lstm_train.py

# GRU
python train/gru_train.py

# Transformer
python train/transformer_train.py
```

Each script:
1. Loads `data/ADBL.csv`
2. Builds 30-day sequence windows
3. Scales features and targets
4. Trains for 300 epochs (Adam, lr=1e-4, MSE loss)
5. Saves model weights to `trained_models/`

### Evaluate and compare

```bash
python evaluation/evaluation.py
```

This loads all three trained models, generates predictions on the test set, and prints:
- Price MAE / RMSE for each model
- Directional accuracy (% of correct up/down predictions)
- Winner by each metric

Outputs `lstm_gru_transformer_comparison.png` with actual vs predicted prices.

## Data

- **File**: `data/ADBL.csv`
- **Rows**: ~3,600
- **Features used**: `open`, `high`, `low`, `traded_quantity`, `traded_amount`, `return_1`, `return_5`, `return_10`
- **Target**: Next-day percentage return (`close.pct_change().shift(-1)`)
- **Split**: Train (< 2026-01-01), Test (>= 2026-01-01)

## Notes

- Keep `shuffle=False` in DataLoaders to preserve temporal order
- Models expect 3D input: `[batch, seq_len, features]`
- 2D inputs are automatically unsqueezed to `[batch, 1, features]`
