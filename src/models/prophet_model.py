"""
prophet_model.py — Facebook Prophet model for coal demand forecasting.
Includes yearly/weekly seasonality and Indian holiday effects.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any

warnings.filterwarnings("ignore")

try:
    from src.config import (
        PROPHET_MODEL_PATH, MODELS_DIR, PROPHET_FORECAST_PATH, REPORTS_DIR,
        PROPHET_YEARLY_SEASONALITY, PROPHET_WEEKLY_SEASONALITY,
        PROCESSED_TRAIN_FILE, PROCESSED_VAL_FILE, PROCESSED_TEST_FILE,
    )
    from src.logger import get_logger
except ImportError:
    from config import (
        PROPHET_MODEL_PATH, MODELS_DIR, PROPHET_FORECAST_PATH, REPORTS_DIR,
        PROPHET_YEARLY_SEASONALITY, PROPHET_WEEKLY_SEASONALITY,
        PROCESSED_TRAIN_FILE, PROCESSED_VAL_FILE, PROCESSED_TEST_FILE,
    )
    from logger import get_logger

logger = get_logger("training")

# Indian holidays for Prophet
INDIAN_HOLIDAYS_PROPHET: list = [
    {"holiday": "Republic Day", "ds": "2022-01-26", "lower_window": -1, "upper_window": 1},
    {"holiday": "Republic Day", "ds": "2023-01-26", "lower_window": -1, "upper_window": 1},
    {"holiday": "Republic Day", "ds": "2024-01-26", "lower_window": -1, "upper_window": 1},
    {"holiday": "Holi", "ds": "2022-03-18", "lower_window": -1, "upper_window": 1},
    {"holiday": "Holi", "ds": "2023-03-08", "lower_window": -1, "upper_window": 1},
    {"holiday": "Holi", "ds": "2024-03-25", "lower_window": -1, "upper_window": 1},
    {"holiday": "Independence Day", "ds": "2022-08-15", "lower_window": -1, "upper_window": 1},
    {"holiday": "Independence Day", "ds": "2023-08-15", "lower_window": -1, "upper_window": 1},
    {"holiday": "Independence Day", "ds": "2024-08-15", "lower_window": -1, "upper_window": 1},
    {"holiday": "Gandhi Jayanti", "ds": "2022-10-02", "lower_window": 0, "upper_window": 0},
    {"holiday": "Gandhi Jayanti", "ds": "2023-10-02", "lower_window": 0, "upper_window": 0},
    {"holiday": "Gandhi Jayanti", "ds": "2024-10-02", "lower_window": 0, "upper_window": 0},
    {"holiday": "Diwali", "ds": "2022-10-24", "lower_window": -2, "upper_window": 2},
    {"holiday": "Diwali", "ds": "2023-11-12", "lower_window": -2, "upper_window": 2},
    {"holiday": "Diwali", "ds": "2024-11-01", "lower_window": -2, "upper_window": 2},
    {"holiday": "Dussehra", "ds": "2022-10-05", "lower_window": -1, "upper_window": 1},
    {"holiday": "Dussehra", "ds": "2023-10-24", "lower_window": -1, "upper_window": 1},
    {"holiday": "Dussehra", "ds": "2024-10-12", "lower_window": -1, "upper_window": 1},
    {"holiday": "Christmas", "ds": "2022-12-25", "lower_window": -1, "upper_window": 1},
    {"holiday": "Christmas", "ds": "2023-12-25", "lower_window": -1, "upper_window": 1},
    {"holiday": "Christmas", "ds": "2024-12-25", "lower_window": -1, "upper_window": 1},
]


def train_prophet(train_df: pd.DataFrame, val_df: pd.DataFrame,
                  test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train Prophet model and forecast on the test period.

    Args:
        train_df: Training data.
        val_df: Validation data.
        test_df: Test data.

    Returns:
        Dict with predictions, actuals, model, and timing info.
    """
    logger.info("=" * 50)
    logger.info("Training Prophet model...")
    logger.info("=" * 50)

    start_time = time.time()

    try:
        from prophet import Prophet  # type: ignore[import-untyped]

        # Combine train + val
        train_val = pd.concat([train_df, val_df], ignore_index=True)

        # Prophet requires columns 'ds' and 'y'
        prophet_train = pd.DataFrame({
            "ds": pd.to_datetime(train_val["date"]),
            "y": train_val["coal_consumption_tonnes"].values,
        })

        test_dates = pd.to_datetime(test_df["date"])
        y_test = test_df["coal_consumption_tonnes"].values
        n_test = len(y_test)

        logger.info(f"Training samples: {len(prophet_train)}, Test samples: {n_test}")

        # Holiday DataFrame
        holidays_df = pd.DataFrame(INDIAN_HOLIDAYS_PROPHET)
        holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])

        # Create and fit model
        model = Prophet(
            yearly_seasonality=PROPHET_YEARLY_SEASONALITY,
            weekly_seasonality=PROPHET_WEEKLY_SEASONALITY,
            daily_seasonality=False,
            holidays=holidays_df,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            interval_width=0.95,
        )

        model.fit(prophet_train)

        # Create future DataFrame for test period
        future = pd.DataFrame({"ds": test_dates})

        # Forecast
        inference_start = time.time()
        forecast = model.predict(future)
        inference_time = (time.time() - inference_start) * 1000  # ms

        predictions = forecast["yhat"].values
        lower_bound = forecast["yhat_lower"].values
        upper_bound = forecast["yhat_upper"].values

        training_time = time.time() - start_time

        # Plot forecast
        os.makedirs(REPORTS_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(test_dates, y_test, label="Actual", color="#2C3E50", linewidth=2)
        ax.plot(test_dates, predictions, label="Prophet Forecast",
                color="#E74C3C", linewidth=2, linestyle="--")
        ax.fill_between(test_dates, lower_bound, upper_bound,
                        alpha=0.2, color="#E74C3C", label="95% CI")
        ax.set_title("Prophet Forecast — Coal Consumption", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Coal Consumption (tonnes)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(PROPHET_FORECAST_PATH, dpi=150)
        plt.close(fig)
        logger.info(f"Prophet forecast plot saved to {PROPHET_FORECAST_PATH}")

        # Save model
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(PROPHET_MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Prophet model saved to {PROPHET_MODEL_PATH}")

        logger.info(f"Prophet training completed in {training_time:.2f}s")
        logger.info(f"Prophet inference time: {inference_time:.2f}ms for {n_test} samples")

        return {
            "model_name": "Prophet",
            "predictions": predictions,
            "actuals": y_test,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "model": model,
            "training_time": training_time,
            "inference_time_ms": inference_time,
        }

    except Exception as e:
        logger.error(f"Prophet training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    train_df = pd.read_csv(PROCESSED_TRAIN_FILE)
    val_df = pd.read_csv(PROCESSED_VAL_FILE)
    test_df = pd.read_csv(PROCESSED_TEST_FILE)
    results = train_prophet(train_df, val_df, test_df)
    print(f"Prophet predictions shape: {results['predictions'].shape}")
