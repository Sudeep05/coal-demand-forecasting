"""
data_generator.py — Generates 3 years of realistic synthetic daily coal demand data.
Includes seasonal patterns, weekday/weekend variation, festival dips,
controlled missing values, and outlier injection.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple

# Import config and logger
try:
    from src.config import (
        DATA_RAW_DIR, RAW_DATA_FILE, DATA_START_DATE, DATA_END_DATE,
        BASE_POWER_MW, EFFICIENCY_FACTOR, MISSING_VALUE_FRACTION,
        OUTLIER_FRACTION, COAL_PRICE_MIN, COAL_PRICE_MAX,
    )
    from src.logger import get_logger
except ImportError:
    from config import (
        DATA_RAW_DIR, RAW_DATA_FILE, DATA_START_DATE, DATA_END_DATE,
        BASE_POWER_MW, EFFICIENCY_FACTOR, MISSING_VALUE_FRACTION,
        OUTLIER_FRACTION, COAL_PRICE_MIN, COAL_PRICE_MAX,
    )
    from logger import get_logger

logger = get_logger("data")

# Indian public holidays (approximate fixed dates)
INDIAN_HOLIDAYS: list = [
    # (month, day, name)
    (1, 26, "Republic Day"),
    (3, 8, "Maha Shivaratri"),
    (3, 25, "Holi"),
    (4, 14, "Ambedkar Jayanti"),
    (5, 1, "May Day"),
    (8, 15, "Independence Day"),
    (8, 19, "Janmashtami"),
    (9, 7, "Ganesh Chaturthi"),
    (10, 2, "Gandhi Jayanti"),
    (10, 15, "Dussehra"),
    (10, 24, "Diwali"),
    (11, 1, "Diwali Holiday"),
    (11, 15, "Guru Nanak Jayanti"),
    (12, 25, "Christmas"),
]


def _get_holiday_flags(dates: pd.DatetimeIndex) -> pd.Series:
    """Mark Indian public holidays across all years in the date range."""
    holiday_set = set()
    for year in dates.year.unique():
        for month, day, _ in INDIAN_HOLIDAYS:
            try:
                holiday_set.add(pd.Timestamp(year=year, month=month, day=day))
            except ValueError:
                pass
    return dates.to_series().apply(lambda d: 1 if d in holiday_set else 0).values


def _generate_temperature(dates: pd.DatetimeIndex) -> np.ndarray:
    """Generate realistic Indian temperature curve with seasonal variation."""
    n = len(dates)
    day_of_year = dates.dayofyear.values.astype(float)
    # Seasonal component: peak around May (day ~140), low around Jan (day ~15)
    seasonal = 25 + 15 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    # Daily random variation
    daily_noise = np.random.normal(0, 2.5, n)
    temperature = seasonal + daily_noise
    return np.clip(temperature, 5.0, 48.0)


def _generate_coal_price(n: int) -> np.ndarray:
    """Generate coal prices with upward trend and volatility."""
    base_price = COAL_PRICE_MIN
    trend = np.linspace(0, COAL_PRICE_MAX - COAL_PRICE_MIN, n) * 0.6
    noise = np.cumsum(np.random.normal(0, 50, n))
    seasonality = 500 * np.sin(2 * np.pi * np.arange(n) / 365)
    price = base_price + trend + noise + seasonality
    return np.clip(price, COAL_PRICE_MIN, COAL_PRICE_MAX)


def _generate_power_generation(dates: pd.DatetimeIndex, is_holiday: np.ndarray) -> np.ndarray:
    """Generate power generation with seasonal peaks and weekday/weekend variation."""
    n = len(dates)
    day_of_year = dates.dayofyear.values.astype(float)

    # Summer peak (May–Jun) and winter peak (Dec–Jan)
    summer_peak = 120 * np.exp(-0.5 * ((day_of_year - 150) / 40) ** 2)
    winter_peak = 80 * np.exp(-0.5 * ((day_of_year - 15) / 30) ** 2)
    winter_peak2 = 80 * np.exp(-0.5 * ((day_of_year - 350) / 30) ** 2)

    # Weekday/weekend variation
    weekday_factor = np.where(dates.dayofweek < 5, 1.0, 0.85)

    # Holiday dip
    holiday_factor = np.where(is_holiday == 1, 0.70, 1.0)

    # Base + seasonal
    power = (BASE_POWER_MW + summer_peak + winter_peak + winter_peak2) * weekday_factor * holiday_factor

    # Random daily noise
    noise = np.random.normal(0, 15, n)
    power = power + noise

    return np.clip(power, 200, 800)


def _inject_missing_values(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Randomly inject NaN values into numeric columns."""
    np.random.seed(42)
    cols_to_corrupt = ["power_generation_mw", "coal_consumption_tonnes",
                       "temperature_c", "coal_price_inr", "inventory_level_tonnes"]
    n_total = len(df)
    for col in cols_to_corrupt:
        if col in df.columns:
            n_missing = int(n_total * fraction / len(cols_to_corrupt))
            idx = np.random.choice(df.index, size=n_missing, replace=False)
            df.loc[idx, col] = np.nan
    return df


def _inject_outliers(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Inject outlier spikes into numeric columns."""
    np.random.seed(99)
    cols = ["coal_consumption_tonnes", "power_generation_mw"]
    n_total = len(df)
    for col in cols:
        if col in df.columns:
            n_outliers = int(n_total * fraction / len(cols))
            idx = np.random.choice(df.index, size=n_outliers, replace=False)
            mean_val = df[col].mean()
            df.loc[idx, col] = mean_val * np.random.uniform(2.0, 3.5, n_outliers)
    return df


def generate_data() -> pd.DataFrame:
    """
    Generate 3 years of realistic synthetic coal demand data.

    Returns:
        pd.DataFrame: Generated dataset with all features.
    """
    logger.info("Starting synthetic data generation...")

    try:
        # Create date range
        dates = pd.date_range(start=DATA_START_DATE, end=DATA_END_DATE, freq="D")
        n = len(dates)
        logger.info(f"Generating {n} days of data from {DATA_START_DATE} to {DATA_END_DATE}")

        # Generate features
        is_holiday = _get_holiday_flags(dates)
        is_weekend = (dates.dayofweek >= 5).astype(int)
        temperature = _generate_temperature(dates)
        coal_price = _generate_coal_price(n)
        power_gen = _generate_power_generation(dates, is_holiday)

        # Coal consumption derived from power generation
        np.random.seed(77)
        efficiency_noise = np.random.normal(EFFICIENCY_FACTOR, 0.03, n)
        coal_consumption = power_gen * np.clip(efficiency_noise, 0.35, 0.55)
        coal_consumption = np.clip(coal_consumption, 80, 500)

        # Inventory level with reorder logic
        inventory = np.zeros(n)
        inventory[0] = 5000.0  # Initial stock
        reorder_point = 2000.0
        reorder_qty = 3000.0
        for i in range(1, n):
            delivery = reorder_qty if inventory[i - 1] < reorder_point else 0
            inventory[i] = inventory[i - 1] - coal_consumption[i] + delivery
            inventory[i] = max(inventory[i], 0)

        # Build DataFrame
        df = pd.DataFrame({
            "date": dates,
            "power_generation_mw": np.round(power_gen, 2),
            "coal_consumption_tonnes": np.round(coal_consumption, 2),
            "temperature_c": np.round(temperature, 1),
            "coal_price_inr": np.round(coal_price, 2),
            "inventory_level_tonnes": np.round(inventory, 2),
            "is_holiday": is_holiday,
            "is_weekend": is_weekend,
        })

        logger.info(f"Raw data shape: {df.shape}")

        # Inject missing values
        df = _inject_missing_values(df, MISSING_VALUE_FRACTION)
        missing_pct = df.isnull().sum() / len(df) * 100
        logger.info("Missing values per column (%):")
        for col, pct in missing_pct.items():
            if pct > 0:
                logger.info(f"  {col}: {pct:.2f}%")

        # Inject outliers
        df = _inject_outliers(df, OUTLIER_FRACTION)
        logger.info(f"Injected ~{OUTLIER_FRACTION*100:.0f}% outliers into key columns")

        # Save to CSV
        os.makedirs(DATA_RAW_DIR, exist_ok=True)
        df.to_csv(RAW_DATA_FILE, index=False)
        logger.info(f"Data saved to {RAW_DATA_FILE}")

        # Log summary statistics
        logger.info("Column statistics:")
        for col in df.select_dtypes(include=[np.number]).columns:
            logger.info(
                f"  {col}: mean={df[col].mean():.2f}, "
                f"std={df[col].std():.2f}, "
                f"min={df[col].min():.2f}, "
                f"max={df[col].max():.2f}"
            )

        logger.info(f"Data generation complete. {len(df)} rows saved.")
        return df

    except Exception as e:
        logger.error(f"Data generation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    df = generate_data()
    print(f"\nGenerated DataFrame shape: {df.shape}")
    print(df.head())
