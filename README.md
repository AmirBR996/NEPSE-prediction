# Stock Prediction Platform

FastAPI backend, static frontend, NEPSE scraping utilities, and PyTorch training pipelines for stock price prediction.

## Overview

This repository combines three parts:

1. A FastAPI backend for authentication, data scraping, and model retraining.
2. A static browser UI for viewing market data and triggering prediction workflows.
3. ML training code for LSTM, GRU, and Transformer models trained on NEPSE historical data.

## Repository Layout

```
stock/
├── backend/
│   ├── auth.py
│   ├── database.py
│   ├── main.py
│   ├── model.py
│   └── routes/
│       ├── auth.py
│       ├── scrape.py
│       └── train.py
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── style.css
├── scraper/
│   ├── nepse_data_scraper.py
│   ├── filter_data.py
│   ├── config/
│   ├── constants/
│   └── utils/
├── ml/
│   ├── data_preprocessing/
│   ├── model/
│   ├── model_evaluation/
│   └── train_pipeline/
├── data/
├── scaler_files/
├── trained_models/
└── requirement.txt
```

## What It Does

- Authenticates users with JWT-backed login and registration.
- Scrapes historical NEPSE data and stores it locally.
- Retrains LSTM, GRU, or Transformer models from the backend.
- Serves a browser UI for dashboard and prediction workflows.
- Stores trained weights and scaler artifacts per company/model.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

The project uses Python 3.14 in the current workspace.

## Run The Backend

Start the API from the repository root:

```bash
uvicorn backend.main:app --reload
```

The backend currently exposes these route groups:

- `GET /` health check
- `/auth` register, login, me, admin
- `/scrape` trigger NEPSE scraping
- `/train` trigger model training

## Run The Frontend

The frontend is a static HTML/CSS/JS app in `frontend/`.

Open `frontend/index.html` with a static server or a live preview extension. If the frontend is served from a different origin, set `API_BASE` in `frontend/app.js` to the backend URL.

## Training And Data

The repository contains pre-trained model artifacts in `trained_models/` and scaler files in both `scaler_files/` and `ml/scaler_files/`.

The training pipeline under `ml/train_pipeline/` includes:

- `lstm_train.py`
- `gru_train.py`
- `transformer_train.py`

The backend training route maps directly to those entry points and accepts a company symbol from:

- `CHCL`
- `CZBIL`
- `BPCL`
- `AHPC`
- `ADBL`
- `ALICL`
- `EBL`
- `NTC`
- `NABIL`
- `PCBL`

## Notes

- The repo contains historical CSV data under `data/` for the listed companies.
- Keep sequence-based ML inputs aligned across preprocessing, training, and evaluation.
- The frontend currently references dashboard and prediction API calls that are not implemented in the checked-in backend routes yet.
