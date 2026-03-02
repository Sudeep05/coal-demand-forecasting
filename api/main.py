"""
main.py — FastAPI application for Coal Demand Forecasting API.
Provides endpoints for prediction, health check, model info, and metrics.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd

from api.schemas import (
    ForecastRequest, ForecastResponse, BatchForecastRequest,
    BatchForecastResponse, ModelInfoResponse, HealthResponse, MetricsResponse,
)
from api.predictor import CoalDemandPredictor

from src.config import (  # type: ignore[import-untyped]
    API_VERSION, MAX_BATCH_SIZE, BEST_MODEL_META_PATH,
    MODEL_COMPARISON_CSV,
)
from src.logger import get_logger  # type: ignore[import-untyped]

logger = get_logger("api")

# ─── App Initialization ────────────────────────────────────────────────────────
app = FastAPI(
    title="Coal Demand Forecasting API",
    description=(
        "ML-powered API for forecasting coal demand in thermal power generation. "
        "Built as part of Masters in Business Analytics Foundation Project."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor (loads model on startup)
predictor: Optional[CoalDemandPredictor] = None


@app.on_event("startup")
async def startup_event() -> None:
    """Load model on application startup."""
    global predictor
    logger.info("Starting Coal Demand Forecasting API...")
    try:
        predictor = CoalDemandPredictor()
        logger.info(f"API started successfully. Model: {predictor.model_name}")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        predictor = None


# ─── Request Logging Middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.2f}ms"
    )
    return response


# ─── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
async def root() -> dict:
    """Root endpoint — API status."""
    model_name = predictor.model_name if predictor else "not loaded"
    return {
        "status": "running",
        "model": model_name,
        "version": API_VERSION,
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    if predictor and predictor.is_loaded:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name=predictor.model_name,
            version=API_VERSION,
        )
    return HealthResponse(
        status="unhealthy",
        model_loaded=False,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info() -> ModelInfoResponse:
    """Get model information and metrics."""
    try:
        if os.path.exists(BEST_MODEL_META_PATH):
            with open(BEST_MODEL_META_PATH, "r") as f:
                meta = json.load(f)
            return ModelInfoResponse(
                model_name=meta.get("model_name", "Unknown"),
                mape=meta.get("mape", 0.0),
                rmse=meta.get("rmse", 0.0),
                trained_on=meta.get("trained_on", "Unknown"),
                version=meta.get("version", API_VERSION),
            )
        raise HTTPException(status_code=404, detail="Model metadata not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=ForecastResponse, tags=["Prediction"])
async def predict(request: ForecastRequest) -> ForecastResponse:
    """Generate a single coal demand forecast."""
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not available")
    try:
        result = predictor.run_prediction(request)
        logger.info(
            f"Prediction: date={request.date}, "
            f"predicted={result.predicted_coal_tonnes:.2f} tonnes"
        )
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchForecastResponse, tags=["Prediction"])
async def predict_batch(request: BatchForecastRequest) -> BatchForecastResponse:
    """Generate batch coal demand forecasts (max 30 days)."""
    if not predictor or not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not available")

    if len(request.requests) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum of {MAX_BATCH_SIZE}"
        )

    try:
        forecasts = predictor.run_batch_prediction(request.requests)
        logger.info(f"Batch prediction: {len(forecasts)} forecasts generated")
        return BatchForecastResponse(
            forecasts=forecasts,
            count=len(forecasts),
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse, tags=["Model"])
async def get_metrics() -> MetricsResponse:
    """Get current model performance metrics."""
    try:
        if os.path.exists(BEST_MODEL_META_PATH):
            with open(BEST_MODEL_META_PATH, "r") as f:
                meta = json.load(f)

            # Try to load detailed metrics from comparison CSV
            mae = None
            r2 = None
            if os.path.exists(MODEL_COMPARISON_CSV):
                comparison = pd.read_csv(MODEL_COMPARISON_CSV)
                model_row = comparison[comparison["Model"] == meta.get("model_name")]
                if not model_row.empty:
                    mae = float(model_row["MAE"].values[0])
                    r2 = float(model_row["R²"].values[0])

            return MetricsResponse(
                model_name=meta.get("model_name", "Unknown"),
                mape=meta.get("mape", 0.0),
                rmse=meta.get("rmse", 0.0),
                mae=mae,
                r2=r2,
                last_updated=meta.get("trained_on"),
            )

        raise HTTPException(status_code=404, detail="Metrics not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
