"""Shared paths for evaluation vs production model artifacts.

Evaluation models: trained with held-out test split (from TEST_SPLIT_DATE).
Production models: trained on all available data; used for live predictions.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Held-out evaluation period starts on this date (inclusive).
TEST_SPLIT_DATE = "2025-01-01"

PURPOSE_EVALUATION = "evaluation"
PURPOSE_PRODUCTION = "production"


def company_model_dir(symbol: str, purpose: str) -> Path:
    if purpose not in (PURPOSE_EVALUATION, PURPOSE_PRODUCTION):
        raise ValueError(f"Unsupported purpose: {purpose}")
    return PROJECT_ROOT / "trained_models" / symbol.upper() / purpose


def model_weights_path(symbol: str, model_name: str, purpose: str) -> Path:
    return company_model_dir(symbol, purpose) / f"stock_{model_name}.pth"


def scaler_paths(symbol: str, purpose: str) -> tuple[Path, Path]:
    base = company_model_dir(symbol, purpose)
    return base / "scaler_X.joblib", base / "scaler_y.joblib"


def model_exists(symbol: str, model_name: str, purpose: str) -> bool:
    if model_weights_path(symbol, model_name, purpose).exists():
        return True
    if purpose == PURPOSE_EVALUATION:
        return legacy_model_path(symbol, model_name).exists()
    return False


def legacy_model_path(symbol: str, model_name: str) -> Path:
    """Pre-separation flat path (migrated to evaluation when present)."""
    return PROJECT_ROOT / "trained_models" / symbol.upper() / f"stock_{model_name}.pth"


def resolve_model_path(symbol: str, model_name: str, purpose: str) -> Path:
    path = model_weights_path(symbol, model_name, purpose)
    if path.exists():
        return path
    # One-time compatibility: old flat weights count as evaluation artifacts
    if purpose == PURPOSE_EVALUATION:
        legacy = legacy_model_path(symbol, model_name)
        if legacy.exists():
            return legacy
    raise FileNotFoundError(
        f"No {purpose} model found for {symbol}/{model_name}: {path}"
    )


def resolve_scaler_paths(symbol: str, purpose: str) -> tuple[Path, Path]:
    sx, sy = scaler_paths(symbol, purpose)
    if sx.exists() and sy.exists():
        return sx, sy
    if purpose == PURPOSE_EVALUATION:
        legacy_dir = PROJECT_ROOT / "trained_models" / symbol.upper()
        lsx, lsy = legacy_dir / "scaler_X.joblib", legacy_dir / "scaler_y.joblib"
        if lsx.exists() and lsy.exists():
            return lsx, lsy
    raise FileNotFoundError(
        f"No {purpose} scalers found for {symbol}"
    )
