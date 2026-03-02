"""
schemas.py — Pydantic models for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date


class ForecastRequest(BaseModel):
    """Single forecast request input."""
    date: str = Field(..., description="Forecast date (YYYY-MM-DD)", example="2025-01-15")
    temperature_c: float = Field(..., description="Temperature in Celsius", example=28.5)
    is_holiday: bool = Field(False, description="Whether date is a holiday")
    is_weekend: bool = Field(False, description="Whether date is a weekend")
    coal_price_inr: float = Field(..., description="Coal price in INR/tonne", example=11000.0)
    power_generation_mw: float = Field(..., description="Power generation in MW", example=520.0)


class ForecastResponse(BaseModel):
    """Single forecast response."""
    forecast_date: str = Field(..., description="Date of forecast")
    predicted_coal_tonnes: float = Field(..., description="Predicted coal consumption in tonnes")
    lower_bound: float = Field(..., description="Lower bound of prediction interval")
    upper_bound: float = Field(..., description="Upper bound of prediction interval")
    confidence_pct: float = Field(..., description="Confidence percentage")
    model_used: str = Field(..., description="Name of model used")


class BatchForecastRequest(BaseModel):
    """Batch forecast request (max 30 days)."""
    requests: List[ForecastRequest] = Field(..., description="List of forecast requests",
                                              max_length=30)


class BatchForecastResponse(BaseModel):
    """Batch forecast response."""
    forecasts: List[ForecastResponse]
    count: int


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    mape: float
    rmse: float
    trained_on: str
    version: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    version: Optional[str] = None


class MetricsResponse(BaseModel):
    """Model metrics response."""
    model_name: str
    mape: float
    rmse: float
    mae: Optional[float] = None
    r2: Optional[float] = None
    last_updated: Optional[str] = None
