"""
arima_model.py — ARIMA/SARIMAX model for coal demand forecasting.
Uses pmdarima auto_arima to find optimal (p,d,q)(P,D,Q,m) parameters.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Dict, Any

warnings.filterwarnings("ignore")

try:
    from src.config import (
        ARIMA_MODEL_PATH, ARIMA_SEASONAL_PERIOD, MODELS_DIR,
        PROCESSED_TRAIN_FILE, PROCESSED_VAL_FILE, PROCESSED_TEST_FILE,
    )
    from src.logger import get_logger
except ImportError:
    from config import (
        ARIMA_MODEL_PATH, ARIMA_SEASONAL_PERIOD, MODELS_DIR,
        PROCESSED_TRAIN_FILE, PROCESSED_VAL_FILE, PROCESSED_TEST_FILE,
    )
    from logger import get_logger

logger = get_logger("training")


def train_arima(train_df: pd.DataFrame, val_df: pd.DataFrame,
                test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train ARIMA model using auto_arima and forecast on the test period.

    Args:
        train_df: Training data.
        val_df: Validation data.
        test_df: Test data.

    Returns:
        Dict with predictions, actuals, model, and timing info.
    """
    logger.info("=" * 50)
    logger.info("Training ARIMA model...")
    logger.info("=" * 50)

    start_time = time.time()

    try:
        import pmdarima as pm  # type: ignore[import-untyped]

        # Combine train + val for final model fitting
        train_val = pd.concat([train_df, val_df], ignore_index=True)
        y_train = train_val["coal_consumption_tonnes"].values
        y_test = test_df["coal_consumption_tonnes"].values
        n_test = len(y_test)

        logger.info(f"Training samples: {len(y_train)}, Test samples: {n_test}")

        # Auto ARIMA to find best parameters
        logger.info("Running auto_arima (this may take a few minutes)...")
        model = pm.auto_arima(
            y_train,
            start_p=0, start_q=0,
            max_p=5, max_q=5, max_d=2,
            seasonal=True, m=ARIMA_SEASONAL_PERIOD,
            start_P=0, start_Q=0,
            max_P=3, max_Q=3, max_D=1,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
            n_fits=50,
        )

        order = model.order
        seasonal_order = model.seasonal_order
        logger.info(f"Selected ARIMA order: {order}")
        logger.info(f"Selected seasonal order: {seasonal_order}")
        logger.info(f"AIC: {model.aic():.2f}")

        # Forecast test period
        inference_start = time.time()
        predictions = model.predict(n_periods=n_test)
        inference_time = (time.time() - inference_start) * 1000  # ms

        # Confidence intervals
        _, conf_int = model.predict(n_periods=n_test, return_conf_int=True)

        training_time = time.time() - start_time

        # Save model
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(ARIMA_MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"ARIMA model saved to {ARIMA_MODEL_PATH}")

        logger.info(f"ARIMA training completed in {training_time:.2f}s")
        logger.info(f"ARIMA inference time: {inference_time:.2f}ms for {n_test} samples")

        return {
            "model_name": "ARIMA",
            "predictions": predictions,
            "actuals": y_test,
            "lower_bound": conf_int[:, 0],
            "upper_bound": conf_int[:, 1],
            "model": model,
            "training_time": training_time,
            "inference_time_ms": inference_time,
            "order": order,
            "seasonal_order": seasonal_order,
        }

    except Exception as e:
        logger.error(f"ARIMA training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    train_df = pd.read_csv(PROCESSED_TRAIN_FILE)
    val_df = pd.read_csv(PROCESSED_VAL_FILE)
    test_df = pd.read_csv(PROCESSED_TEST_FILE)
    results = train_arima(train_df, val_df, test_df)
    print(f"ARIMA predictions shape: {results['predictions'].shape}")
