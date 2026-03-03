"""
config.py — Centralized configuration for Coal Demand Forecasting project.
All constants, file paths, model hyperparameters, and thresholds.
No hardcoded values should exist anywhere else in the codebase.
"""

import os
from pathlib import Path

# ─── PROJECT ROOT ───────────────────────────────────────────────────────────────
PROJECT_ROOT: str = str(Path(__file__).resolve().parent.parent)

# ─── DIRECTORY PATHS ────────────────────────────────────────────────────────────
DATA_RAW_DIR: str = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR: str = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR: str = os.path.join(PROJECT_ROOT, "logs")
REPORTS_DIR: str = os.path.join(PROJECT_ROOT, "reports")

# ─── DATA FILES ─────────────────────────────────────────────────────────────────
RAW_DATA_FILE: str = os.path.join(DATA_RAW_DIR, "coal_data.csv")
PROCESSED_TRAIN_FILE: str = os.path.join(DATA_PROCESSED_DIR, "train.csv")
PROCESSED_VAL_FILE: str = os.path.join(DATA_PROCESSED_DIR, "val.csv")
PROCESSED_TEST_FILE: str = os.path.join(DATA_PROCESSED_DIR, "test.csv")
PROCESSED_FULL_FILE: str = os.path.join(DATA_PROCESSED_DIR, "processed_data.csv")

# ─── MODEL ARTIFACT PATHS ──────────────────────────────────────────────────────
SCALER_PATH: str = os.path.join(MODELS_DIR, "scaler.pkl")
ARIMA_MODEL_PATH: str = os.path.join(MODELS_DIR, "arima_model.pkl")
PROPHET_MODEL_PATH: str = os.path.join(MODELS_DIR, "prophet_model.pkl")
LSTM_MODEL_PATH: str = os.path.join(MODELS_DIR, "lstm_model.pt")
XGBOOST_MODEL_PATH: str = os.path.join(MODELS_DIR, "xgboost_model.pkl")
BEST_MODEL_META_PATH: str = os.path.join(MODELS_DIR, "best_model_meta.json")

# ─── REPORT / PLOT PATHS ───────────────────────────────────────────────────────
SIPOC_DIAGRAM_PATH: str = os.path.join(REPORTS_DIR, "sipoc_diagram.png")
VALUE_CHAIN_PATH: str = os.path.join(REPORTS_DIR, "value_chain.png")
TREND_PLOT_PATH: str = os.path.join(REPORTS_DIR, "trend_analysis.png")
SEASONALITY_PLOT_PATH: str = os.path.join(REPORTS_DIR, "seasonality_decomposition.png")
ACF_PACF_PLOT_PATH: str = os.path.join(REPORTS_DIR, "acf_pacf.png")
CORRELATION_HEATMAP_PATH: str = os.path.join(REPORTS_DIR, "correlation_heatmap.png")
OUTLIER_BOXPLOT_PATH: str = os.path.join(REPORTS_DIR, "outlier_boxplots.png")
PROPHET_FORECAST_PATH: str = os.path.join(REPORTS_DIR, "prophet_forecast.png")
LSTM_LOSS_PATH: str = os.path.join(REPORTS_DIR, "lstm_loss.png")
XGB_FEATURE_IMPORTANCE_PATH: str = os.path.join(REPORTS_DIR, "xgb_feature_importance.png")
MODEL_COMPARISON_CSV: str = os.path.join(REPORTS_DIR, "model_comparison.csv")
BEST_MODEL_FORECAST_PATH: str = os.path.join(REPORTS_DIR, "best_model_forecast.png")
PREDICTION_INTERVALS_PATH: str = os.path.join(REPORTS_DIR, "prediction_intervals.png")

# ─── DATA GENERATION PARAMETERS ────────────────────────────────────────────────
DATA_START_DATE: str = "2022-01-01"
DATA_END_DATE: str = "2024-12-31"
BASE_POWER_MW: float = 500.0
EFFICIENCY_FACTOR: float = 0.45
MISSING_VALUE_FRACTION: float = 0.05
OUTLIER_FRACTION: float = 0.03
COAL_PRICE_MIN: float = 8000.0
COAL_PRICE_MAX: float = 14000.0

# ─── PREPROCESSING PARAMETERS ──────────────────────────────────────────────────
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15
LAG_FEATURES: list = [1, 2, 3, 7, 14, 30]
ROLLING_WINDOWS: list = [7, 14, 30]
IQR_MULTIPLIER: float = 1.5

# ─── MODEL HYPERPARAMETERS ─────────────────────────────────────────────────────
# ARIMA
ARIMA_SEASONAL_PERIOD: int = 12

# LSTM
LSTM_SEQUENCE_LENGTH: int = 30
LSTM_LAYER_1_UNITS: int = 128
LSTM_LAYER_2_UNITS: int = 64
LSTM_DROPOUT_RATE: float = 0.2
LSTM_EPOCHS: int = 100
LSTM_BATCH_SIZE: int = 32
LSTM_EARLY_STOP_PATIENCE: int = 10

# XGBoost / Optuna
OPTUNA_N_TRIALS: int = 100

# Prophet
PROPHET_YEARLY_SEASONALITY: bool = True
PROPHET_WEEKLY_SEASONALITY: bool = True

# ─── EVALUATION & MONITORING THRESHOLDS ─────────────────────────────────────────
TARGET_MAPE: float = 8.0        # Target: MAPE < 8%
TARGET_RMSE: float = 50.0       # Target: RMSE < 50 tonnes
RETRAIN_MAPE_THRESHOLD: float = 10.0   # MAPE > 10% triggers retrain alert
DRIFT_SIGNIFICANCE: float = 0.05       # KS-test p-value threshold

# ─── ECONOMIC COST ASSUMPTIONS (INR) ───────────────────────────────────────────
HOLDING_COST_PER_TONNE_PER_DAY: float = 500.0      # Rs 500/tonne/day
SHORTAGE_COST_PER_MWH: float = 8000.0               # Rs 8000/MWh
AVG_MWH_LOST_PER_SHORTAGE: float = 200.0            # Average MWh lost per event
INVENTORY_REDUCTION_PERCENT: float = 15.0            # Expected inventory reduction %
DAYS_PER_YEAR: int = 365

# ─── API SETTINGS ──────────────────────────────────────────────────────────────
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000
API_VERSION: str = "1.0"
MAX_BATCH_SIZE: int = 30
CONFIDENCE_PERCENT: float = 95.0

# ─── LOGGING SETTINGS ──────────────────────────────────────────────────────────
LOG_MAX_BYTES: int = 10 * 1024 * 1024   # 10 MB
LOG_BACKUP_COUNT: int = 5
LOG_FORMAT: str = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
