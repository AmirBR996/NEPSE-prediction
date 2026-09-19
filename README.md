# StockPulse

Stock forecasting platform with JWT authentication, admin panel, and ML pipelines for NEPSE (Nepal Stock Exchange) price prediction.

## Overview

StockPulse combines three systems:

- **FastAPI backend** — authentication, company data management, ML model training/evaluation, and stock predictions.
- **Static frontend** — self-contained HTML pages (CSS + JS inlined) for dashboard, auth, predictions, and admin.
- **ML pipelines** — LSTM, GRU, and Transformer models trained on NEPSE historical data, with evaluation and production model management.

## Architecture

```
stock/
├── backend/            # FastAPI application
│   ├── main.py         # App entry point (routes, middleware, frontend serving)
│   ├── auth.py         # JWT token creation, password hashing
│   ├── database.py     # SQLite connection, session factory, declarative base
│   ├── deps.py         # Dependency injections (auth guards, user resolution)
│   ├── model.py        # SQLAlchemy models (User, Company, ModelRecord, TrainingJob, ModelEvaluation, Prediction)
│   ├── constants.py    # Company symbols, model types, roles, admin credentials
│   ├── seed.py         # Database seeding & schema migration
│   ├── routes/         # API routers
│   │   ├── auth.py       # POST /register, /login; GET /me, /logout, /admin
│   │   ├── companies.py  # GET /companies, /{id}, /{id}/data, /{id}/forecast
│   │   ├── predictions.py # POST /predictions; GET /predictions, /{id}
│   │   ├── admin.py      # Admin-only: users, companies, models, evaluations, training jobs
│   │   ├── train.py      # User training endpoints (/train/*)
│   │   ├── training_jobs.py # POST /training/user; GET /training/{id}
│   │   ├── scrape.py     # POST /scrape/ — trigger NEPSE data scrape
│   │   └── __init__.py
│   ├── services/       # Business logic
│   │   ├── prediction.py    # predict_next_days, run_company_evaluation, persist_evaluation
│   │   ├── companies.py     # company_overview, company_history, list_companies, set_production_model, latest_evaluations
│   │   ├── training.py      # create/cancel/delete training jobs, start_training_job, list_jobs
│   │   └── __init__.py
│   └── routes/
├── ml/                 # ML code
│   ├── model/              # StockLSTM, StockGRU, StockTransformer (PyTorch)
│   ├── data_preprocessing/feature.py   # Feature engineering, sequence creation, scalers
│   ├── model_evaluation/evaluation.py  # Model evaluation (metrics, price reconstruction)
│   ├── model_paths.py              # Model/scaler artifact paths (evaluation vs production)
│   └── train_pipeline/             # Training entry points (train.py, user_train.py)
├── scraper/            # NEPSE data scraping
│   ├── nepse_data_scraper.py   # Main scraper: fetches historical price data
│   ├── config/headers.py       # HTTP headers for scraping
│   ├── constants/companyIdMap.py  # Company ID mapping for sharesansar.com
│   ├── constants/url.py        # Scraper URLs
│   └── utils/                  # Scraping utilities (session, params, history)
├── frontend/pages/     # Self-contained HTML pages
│   ├── index.html   # Dashboard (market overview, charts, model forecasts)
│   ├── auth.html    # Login / Register
│   ├── predict.html # Predictions & user training
│   └── admin.html   # Admin panel (users, companies, training, evaluation, system)
├── data/                  # Historical CSV data (per-company)
├── trained_models/        # Trained model weights per company/purpose
├── scaler_files/          # Scaler artifacts (legacy)
├── .env
├── requirement.txt
└── README.md
```

## Features

### Authentication
- User registration and login with hashed passwords (argon2)
- JWT-based authentication with configurable expiry
- Role-based access control (ADMIN, USER)
- User authorization system (admin approves users for prediction access)

### Companies
- Browse all supported NEPSE companies with market data
- Historical price data with date range filtering (1M, 3M, 6M, 1Y, 5Y, ALL)
- Company overview with training status, model status, and latest prices
- Forecast overview with evaluation metrics

### Predictions
- Generate stock price predictions using production ML models
- Support for LSTM, GRU, Transformer, or auto-select production model
- Configurable prediction horizon (1-7 days)
- Prediction history per user

### Training (Admin & User)
- Train LSTM, GRU, or Transformer models
- Evaluation training (held-out test split from 2025) and Production training (all data)
- Per-company, per-model training job management
- Training progress tracking with real-time status
- Production model deployment after evaluation
- Stale job detection and cleanup

### Admin Panel
- User management (authorize, activate/deactivate, delete)
- Company management with status filtering
- Model overview and evaluation metrics
- Training job management (view, cancel, delete, clear stale/failed)
- System status dashboard
- Run model evaluation
- Trigger NEPSE data scraping

### Data Scraping
- Automated NEPSE historical price scraping from sharesansar.com
- Incremental updates (fetches only new data since last scrape)
- Configurable via environment variables

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

The project uses Python 3.14+ in the current workspace.

## Configuration

Environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `dev-secret-key-change-me` | JWT signing secret |
| `ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_HOURS` | `12` | Token expiry in hours |

## Running

### Backend

```bash
uvicorn backend.main:app --reload
```

The API is available at `http://localhost:8000` with interactive docs at `/docs`.

### Frontend

The frontend is served by the FastAPI app. Each page is a self-contained HTML file with CSS and JS inlined.

| Page | Route | Purpose |
|------|-------|---------|
| Dashboard | `/` | Market overview, charts, model forecasts |
| Auth | `/auth.html` | Login / Register |
| Predict | `/predict.html` | Forecasts & training UI |
| Admin | `/admin.html` | Admin panel (users, training, evaluation) |

To change the API base URL from the browser, set `localStorage.api_base_url` (default `http://localhost:8000`).

## Supported Companies

| Symbol | Company Name |
|--------|-------------|
| CHCL | Chilime Hydropower Company |
| CZBIL | Citizen Bank International |
| BPCL | Butwal Power Company |
| AHPC | Arun Valley Hydropower |
| ADBL | Agriculture Development Bank |
| ALICL | Asian Life Insurance |
| EBL | Everest Bank |
| NTC | Nepal Doorsanchar (NTC) |
| NABIL | Nabil Bank |
| PCBL | Prime Commercial Bank |

## ML Models

Three model architectures are supported:

| Model | Architecture |
|-------|-------------|
| LSTM | Long Short-Term Memory networks |
| GRU | Gated Recurrent Units |
| Transformer | Encoder-only Transformer with positional encoding |

### Training Pipeline

- **Evaluation training** (`include_test=True`): Trains on data before 2025-01-01, tests on 2025+ data. Used for model comparison.
- **Production training** (`include_test=False`): Trains on all available data. Used for live predictions.

Training artifacts are stored per company and purpose:
- `trained_models/{symbol}/evaluation/` — Evaluation model weights and scalers
- `trained_models/{symbol}/production/` — Production model weights and scalers

## API Reference

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register new user |
| POST | `/auth/login` | Login and get JWT token |
| GET | `/auth/me` | Current user info |
| POST | `/auth/logout` | Logout |
| GET | `/auth/admin` | Admin check |

### Companies
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/companies` | List all companies |
| GET | `/companies/{id}` | Company overview |
| GET | `/companies/{id}/data?range=1Y` | Historical price data |
| GET | `/companies/{id}/forecast` | Forecast overview |

### Predictions
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predictions` | Create prediction |
| GET | `/predictions` | User prediction history |
| GET | `/predictions/{id}` | Get prediction result |

### Training (User)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/train/trainall` | Train all active companies |
| POST | `/train/{model}/{data}` | Train single model |
| POST | `/train/{data}` | Train all models for company |
| POST | `/training/user` | Start user training job |
| GET | `/training/{job_id}` | Get training job status |

### Admin
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/stats` | System statistics |
| GET | `/admin/system` | System status |
| GET | `/admin/users` | List all users |
| PATCH | `/admin/users/{id}/authorize` | Toggle user authorization |
| PATCH | `/admin/users/{id}/status` | Toggle user active status |
| DELETE | `/admin/users/{id}` | Delete user |
| GET | `/admin/companies` | List companies with filters |
| POST | `/admin/companies/{id}/train` | Train company models |
| POST | `/admin/companies/{id}/evaluate` | Evaluate company models |
| GET | `/admin/models` | Model overview |
| GET | `/admin/models/{id}` | Company model details |
| POST | `/admin/models/{id}/production` | Deploy production model |
| POST | `/admin/companies/{id}/production` | Set company production model |
| GET | `/admin/evaluations/{id}` | Model evaluations |
| POST | `/admin/evaluate/{data}` | Run evaluation |
| GET | `/admin/training/jobs` | List training jobs |
| POST | `/admin/training/jobs/clear-stale` | Clear stale jobs |
| POST | `/admin/training/jobs/clear-failed` | Clear failed jobs |
| DELETE | `/admin/training/jobs/{id}` | Delete job |
| POST | `/admin/training/jobs/{id}/cancel` | Cancel job |
| POST | `/admin/trainall` | Train all companies × all models |
| POST | `/scrape/` | Trigger NEPSE data scrape |

## Default Admin Credentials

| Field | Value |
|-------|-------|
| Username | `admin` |
| Email | `admin@stockpulse.local` |
| Password | `admin123` |

**Change these before deploying.**
