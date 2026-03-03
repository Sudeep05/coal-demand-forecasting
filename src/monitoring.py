"""
monitoring.py — Data drift detection and model performance monitoring.
Called by the Streamlit dashboard for live drift checks.
Uses the KS test (Kolmogorov–Smirnov) to detect distributional shifts
between training and test/production data.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import (
    PROCESSED_TRAIN_FILE,
    PROCESSED_TEST_FILE,
    BEST_MODEL_META_PATH,
    XGBOOST_MODEL_PATH,
    SCALER_PATH,
)
from src.logger import get_logger

logger = get_logger(__name__)

# Target column name in processed CSVs
TARGET_COL = "coal_consumption_tonnes"

# Columns to SKIP when checking drift
# — target, date, and engineered features (lags, rolling stats, calendar)
#   because time-ordered splits naturally cause distributional shifts
#   in time-derived features — that's not real data drift.
_SKIP_COLS = {
    "date", "coal_demand_tonnes", "coal_consumption_tonnes", "ds", "y",
    # Calendar features (always differ between time windows)
    "month", "quarter", "day_of_week", "day_of_year", "week_of_year",
    # Lag features (shift with the target)
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30",
    # Rolling / EWM features (derived from target)
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
    "rolling_std_7", "rolling_min_7", "rolling_max_7", "ewm_7",
    # Interaction / derived features
    "power_lag_1", "temp_coal_interaction", "temp_squared",
    "power_temp_interaction",
}


def detect_data_drift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold: float = 0.05,
) -> dict:
    """
    Run a two-sample KS test on each numeric feature.
    Returns a dict with per-feature results and an overall drift flag.
    """
    feature_cols = [
        c for c in train_df.select_dtypes(include=[np.number]).columns
        if c.lower() not in _SKIP_COLS
    ]

    feature_results = {}
    any_drift = False

    for col in feature_cols:
        train_vals = train_df[col].dropna().values
        test_vals = test_df[col].dropna().values

        if len(train_vals) < 5 or len(test_vals) < 5:
            continue

        ks_stat, p_value = ks_2samp(train_vals, test_vals)
        drift_detected = p_value < threshold

        feature_results[col] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "drift_detected": drift_detected,
        }
        if drift_detected:
            any_drift = True

    return {
        "overall_drift": any_drift,
        "threshold": threshold,
        "features_checked": len(feature_results),
        "features_drifted": sum(
            1 for v in feature_results.values() if v["drift_detected"]
        ),
        "feature_results": feature_results,
    }


def check_performance(
    test_df: pd.DataFrame,
    mape_threshold: float = 10.0,
) -> dict:
    """
    Check if current model MAPE exceeds the degradation threshold.
    Uses the best model (XGBoost) for inference on the test set.
    """
    import joblib

    try:
        # Load model
        model = joblib.load(XGBOOST_MODEL_PATH)

        # Use ALL numeric features (same as training) — only exclude date + target
        _perf_skip = {"date", "coal_consumption_tonnes", "coal_demand_tonnes", "ds", "y"}
        feature_cols = [
            c for c in test_df.select_dtypes(include=[np.number]).columns
            if c.lower() not in _perf_skip
        ]

        X_test = test_df[feature_cols]
        y_test = test_df[TARGET_COL].values

        # XGBoost doesn't need scaling — predict directly
        preds = model.predict(X_test)

        mask = y_test != 0
        current_mape = float(
            np.mean(np.abs((y_test[mask] - preds[mask]) / y_test[mask])) * 100
        )

        return {
            "current_mape": current_mape,
            "threshold": mape_threshold,
            "is_degraded": current_mape > mape_threshold,
        }

    except Exception as e:
        logger.error(f"Performance check failed: {e}")
        return {
            "current_mape": None,
            "threshold": mape_threshold,
            "is_degraded": False,
            "error": str(e),
        }


def run_monitoring() -> dict:
    """
    Entry point called by the Streamlit dashboard.
    Loads train/test CSVs, runs drift detection and performance check.
    Returns combined results dict.
    """
    logger.info("Running monitoring pipeline …")

    # Load data
    if not os.path.exists(PROCESSED_TRAIN_FILE):
        raise FileNotFoundError(
            f"Training data not found at {PROCESSED_TRAIN_FILE}. "
            "Run the pipeline notebooks first."
        )

    train_df = pd.read_csv(PROCESSED_TRAIN_FILE)
    test_df = pd.read_csv(PROCESSED_TEST_FILE)

    # Drift detection
    drift_results = detect_data_drift(train_df, test_df)
    logger.info(
        f"Drift check: {drift_results['features_drifted']}/{drift_results['features_checked']} "
        f"features drifted — overall_drift={drift_results['overall_drift']}"
    )

    # Performance check
    perf_results = check_performance(test_df)
    if perf_results.get("current_mape") is not None:
        logger.info(f"Current MAPE: {perf_results['current_mape']:.2f}%")

    return {
        "drift": drift_results,
        "performance": perf_results,
    }


if __name__ == "__main__":
    results = run_monitoring()
    print(json.dumps(results, indent=2, default=str))
