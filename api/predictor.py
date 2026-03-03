"""
predictor.py — Loads best model and scaler, runs inference for API predictions.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (  # type: ignore[import-untyped]
    BEST_MODEL_META_PATH, XGBOOST_MODEL_PATH, ARIMA_MODEL_PATH,
    PROPHET_MODEL_PATH, LSTM_MODEL_PATH, SCALER_PATH,
    CONFIDENCE_PERCENT, MODELS_DIR,
)
from src.logger import get_logger  # type: ignore[import-untyped]

from api.schemas import ForecastRequest, ForecastResponse

logger = get_logger("api")


class CoalDemandPredictor:
    """Loads and serves the best trained model for coal demand prediction."""

    def __init__(self) -> None:
        self.model = None
        self.scaler = None
        self.model_name: str = "Unknown"
        self.meta: Dict[str, Any] = {}
        self.is_loaded: bool = False
        self._load_model()

    def _load_model(self) -> None:
        """Load best model and scaler from disk."""
        try:
            # Load metadata
            if os.path.exists(BEST_MODEL_META_PATH):
                with open(BEST_MODEL_META_PATH, "r") as f:
                    self.meta = json.load(f)
                self.model_name = self.meta.get("model_name", "XGBoost")
            else:
                logger.warning(f"Model metadata not found at {BEST_MODEL_META_PATH}")
                self.model_name = "XGBoost"

            # Load scaler
            if os.path.exists(SCALER_PATH):
                with open(SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info("Scaler loaded successfully")
            else:
                logger.warning(f"Scaler not found at {SCALER_PATH}")

            # Load model based on best model name
            model_path_map = {
                "XGBoost": XGBOOST_MODEL_PATH,
                "ARIMA": ARIMA_MODEL_PATH,
                "Prophet": PROPHET_MODEL_PATH,
            }

            model_path = model_path_map.get(self.model_name, XGBOOST_MODEL_PATH)

            if self.model_name == "LSTM" and os.path.exists(LSTM_MODEL_PATH):
                import torch
                from src.models.lstm_model import CoalLSTM  # type: ignore[import-untyped]
                ckpt = torch.load(LSTM_MODEL_PATH, map_location="cpu", weights_only=False)
                lstm_model = CoalLSTM(
                    n_features=ckpt["n_features"],
                    layer1_units=ckpt["layer1_units"],
                    layer2_units=ckpt["layer2_units"],
                    dropout=ckpt["dropout"],
                )
                lstm_model.load_state_dict(ckpt["model_state_dict"])
                lstm_model.eval()
                self.model = lstm_model
                logger.info(f"LSTM model loaded from {LSTM_MODEL_PATH}")
            elif os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info(f"{self.model_name} model loaded from {model_path}")
            else:
                # Fallback: try loading any available model
                for name, path in model_path_map.items():
                    if os.path.exists(path):
                        with open(path, "rb") as f:
                            self.model = pickle.load(f)
                        self.model_name = name
                        logger.info(f"Fallback: loaded {name} from {path}")
                        break

            if self.model is not None:
                self.is_loaded = True
                logger.info(f"Model '{self.model_name}' ready for serving")
            else:
                logger.error("No model could be loaded!")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self.is_loaded = False

    def _prepare_features(self, request: ForecastRequest) -> np.ndarray:
        """Transform API request into model feature vector (28 features for XGBoost)."""
        try:
            dt = pd.to_datetime(request.date)

            # Estimated demand proxies based on inputs
            demand_est = request.power_generation_mw * 0.45
            demand_est_prev = demand_est * 0.98

            features = {
                # --- 6 raw input features ---
                "power_generation_mw": request.power_generation_mw,
                "temperature_c": request.temperature_c,
                "coal_price_inr": request.coal_price_inr,
                "inventory_level_tonnes": 3000.0,  # default estimate
                "is_holiday": int(request.is_holiday),
                "is_weekend": int(request.is_weekend),
                # --- 5 calendar features ---
                "month": dt.month,
                "quarter": dt.quarter,
                "day_of_week": dt.dayofweek,
                "day_of_year": dt.dayofyear,
                "week_of_year": int(dt.strftime("%W")),
                # --- 6 lag features (estimated from demand proxy) ---
                "lag_1": demand_est,
                "lag_2": demand_est * 0.99,
                "lag_3": demand_est * 0.98,
                "lag_7": demand_est * 0.97,
                "lag_14": demand_est * 0.96,
                "lag_30": demand_est * 0.95,
                # --- 6 rolling / ewm features ---
                "rolling_mean_7": demand_est,
                "rolling_mean_14": demand_est * 0.99,
                "rolling_mean_30": demand_est * 0.98,
                "rolling_std_7": 15.0,
                "rolling_min_7": demand_est * 0.92,
                "rolling_max_7": demand_est * 1.08,
                "ewm_7": demand_est,
                # --- 1 power lag ---
                "power_lag_1": request.power_generation_mw * 0.99,
                # --- 3 interaction / polynomial features ---
                "temp_coal_interaction": request.temperature_c * request.coal_price_inr,
                "temp_squared": request.temperature_c ** 2,
                "power_temp_interaction": request.power_generation_mw * request.temperature_c,
            }

            feature_df = pd.DataFrame([features])

            # XGBoost does NOT need scaling — predict on raw features
            if self.model_name == "XGBoost":
                return feature_df.values

            # For other models that may need scaling
            if self.scaler is not None:
                try:
                    expected_cols = self.scaler.feature_names_in_
                    for col in expected_cols:
                        if col not in feature_df.columns:
                            feature_df[col] = 0.0
                    feature_df = feature_df[expected_cols]
                except AttributeError:
                    pass
                return self.scaler.transform(feature_df)

            return feature_df.values

        except Exception as e:
            logger.error(f"Feature preparation failed: {e}", exc_info=True)
            raise

    def run_prediction(self, request: ForecastRequest) -> ForecastResponse:
        """
        Run single prediction.

        Args:
            request: ForecastRequest with input features.

        Returns:
            ForecastResponse with prediction and confidence bounds.
        """
        try:
            if not self.is_loaded:
                raise RuntimeError("Model not loaded")

            features = self._prepare_features(request)

            if self.model_name == "ARIMA":
                prediction = float(self.model.predict(n_periods=1)[0])
            elif self.model_name == "Prophet":
                future = pd.DataFrame({"ds": [pd.to_datetime(request.date)]})
                forecast = self.model.predict(future)
                prediction = float(forecast["yhat"].values[0])
            else:
                # XGBoost or sklearn-compatible
                prediction = float(self.model.predict(features)[0])

            # Confidence bounds (±5% as approximation)
            margin = abs(prediction) * 0.05
            lower = prediction - margin
            upper = prediction + margin

            return ForecastResponse(
                forecast_date=request.date,
                predicted_coal_tonnes=round(prediction, 2),
                lower_bound=round(lower, 2),
                upper_bound=round(upper, 2),
                confidence_pct=CONFIDENCE_PERCENT,
                model_used=self.model_name,
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise

    def run_batch_prediction(self, requests: List[ForecastRequest]) -> List[ForecastResponse]:
        """
        Run batch predictions.

        Args:
            requests: List of ForecastRequest objects.

        Returns:
            List of ForecastResponse objects.
        """
        logger.info(f"Running batch prediction for {len(requests)} requests")
        results = []
        for req in requests:
            try:
                result = self.run_prediction(req)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction failed for {req.date}: {e}")
                # Return a zero prediction on error
                results.append(ForecastResponse(
                    forecast_date=req.date,
                    predicted_coal_tonnes=0.0,
                    lower_bound=0.0,
                    upper_bound=0.0,
                    confidence_pct=0.0,
                    model_used=self.model_name,
                ))
        return results
