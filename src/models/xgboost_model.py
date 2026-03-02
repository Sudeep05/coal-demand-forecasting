"""
xgboost_model.py — XGBoost model for coal demand forecasting.
Uses lag + rolling + time features with Optuna hyperparameter tuning.
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
        XGBOOST_MODEL_PATH, MODELS_DIR, REPORTS_DIR,
        XGB_FEATURE_IMPORTANCE_PATH, OPTUNA_N_TRIALS,
        PROCESSED_TRAIN_FILE, PROCESSED_VAL_FILE, PROCESSED_TEST_FILE,
    )
    from src.logger import get_logger
except ImportError:
    from config import (
        XGBOOST_MODEL_PATH, MODELS_DIR, REPORTS_DIR,
        XGB_FEATURE_IMPORTANCE_PATH, OPTUNA_N_TRIALS,
        PROCESSED_TRAIN_FILE, PROCESSED_VAL_FILE, PROCESSED_TEST_FILE,
    )
    from logger import get_logger

logger = get_logger("training")


def train_xgboost(train_df: pd.DataFrame, val_df: pd.DataFrame,
                  test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train XGBoost model with Optuna hyperparameter tuning.

    Args:
        train_df: Training data.
        val_df: Validation data.
        test_df: Test data.

    Returns:
        Dict with predictions, actuals, model, and timing info.
    """
    logger.info("=" * 50)
    logger.info("Training XGBoost model...")
    logger.info("=" * 50)

    start_time = time.time()

    try:
        import xgboost as xgb  # type: ignore[import-untyped]
        import optuna  # type: ignore[import-untyped]
        from sklearn.metrics import mean_squared_error

        # Suppress optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Feature columns
        feature_cols = [c for c in train_df.columns
                        if c not in ["date", "coal_consumption_tonnes"]]

        X_train = train_df[feature_cols].values
        y_train = train_df["coal_consumption_tonnes"].values
        X_val = val_df[feature_cols].values
        y_val = val_df["coal_consumption_tonnes"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["coal_consumption_tonnes"].values

        logger.info(f"Training samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Feature names: {feature_cols}")

        # Optuna objective
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            }

            model = xgb.XGBRegressor(
                **params,
                objective="reg:squarederror",
                random_state=42,
                verbosity=0,
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            val_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            return rmse

        # Run Optuna optimization
        logger.info(f"Starting Optuna hyperparameter tuning ({OPTUNA_N_TRIALS} trials)...")
        study = optuna.create_study(direction="minimize", study_name="xgboost_coal")
        study.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)

        best_params = study.best_params
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best validation RMSE: {study.best_value:.4f}")

        # Train final model with best params
        final_model = xgb.XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
        )

        # Combine train + val for final training
        X_train_full = np.concatenate([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])

        final_model.fit(X_train_full, y_train_full, verbose=False)

        # Predict on test
        inference_start = time.time()
        predictions = final_model.predict(X_test)
        inference_time = (time.time() - inference_start) * 1000

        training_time = time.time() - start_time

        # Feature importance plot
        os.makedirs(REPORTS_DIR, exist_ok=True)
        importances = final_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:15]  # Top 15

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(
            [feature_cols[i] for i in sorted_idx][::-1],
            importances[sorted_idx][::-1],
            color="#3498DB",
            edgecolor="#2C3E50",
        )
        ax.set_title("XGBoost — Top Feature Importances", fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        fig.savefig(XGB_FEATURE_IMPORTANCE_PATH, dpi=150)
        plt.close(fig)
        logger.info(f"Feature importance plot saved to {XGB_FEATURE_IMPORTANCE_PATH}")

        # Save model
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(XGBOOST_MODEL_PATH, "wb") as f:
            pickle.dump(final_model, f)
        logger.info(f"XGBoost model saved to {XGBOOST_MODEL_PATH}")

        logger.info(f"XGBoost training completed in {training_time:.2f}s")
        logger.info(f"XGBoost inference time: {inference_time:.2f}ms for {len(predictions)} samples")

        return {
            "model_name": "XGBoost",
            "predictions": predictions,
            "actuals": y_test,
            "model": final_model,
            "training_time": training_time,
            "inference_time_ms": inference_time,
            "best_params": best_params,
            "feature_cols": feature_cols,
        }

    except Exception as e:
        logger.error(f"XGBoost training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    train_df = pd.read_csv(PROCESSED_TRAIN_FILE)
    val_df = pd.read_csv(PROCESSED_VAL_FILE)
    test_df = pd.read_csv(PROCESSED_TEST_FILE)
    results = train_xgboost(train_df, val_df, test_df)
    print(f"XGBoost predictions shape: {results['predictions'].shape}")
