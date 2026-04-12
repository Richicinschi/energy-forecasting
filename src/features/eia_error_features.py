#!/usr/bin/env python3
"""Phase 5: EIA Error Correction Features

Learn systematic biases in EIA's day-ahead forecast using 48h-old data.
All features are leakage-safe (use t-48 or older EIA forecasts).
"""

import numpy as np
import pandas as pd
from sqlalchemy import text


def add_eia_error_features(df: pd.DataFrame, engine, respondent: str) -> pd.DataFrame:
    """Add Phase 5 EIA error correction features.
    
    These features capture systematic biases in EIA's day-ahead forecast
    that we can learn from historical errors.
    
    All EIA-based features use 48h-old data (leakage-safe for 24h-ahead prediction).
    
    Args:
        df: DataFrame with UTC DatetimeIndex, must have eia_forecast_mw column
        engine: SQLAlchemy engine
        respondent: BA code
    
    Returns:
        DataFrame with EIA error features added
    """
    idx = df.index
    
    # We need EIA forecast (DF) and actual demand (D) from t-48 and earlier
    # At prediction time (t-24), we have access to:
    # - EIA forecast made at t-48 for hour t-24 (published next day)
    # - Actual demand at t-48, t-72, etc.
    
    if "eia_forecast_mw" not in df.columns or "lag_24h" not in df.columns:
        # Add placeholder columns
        df["eia_bias_24h"] = np.float32(np.nan)
        df["eia_bias_48h"] = np.float32(np.nan)
        df["eia_forecast_change_24h"] = np.float32(np.nan)
        df["eia_vs_persistence"] = np.float32(np.nan)
        df["eia_confidence"] = np.float32(np.nan)
        return df
    
    # Current values (at t-24 from perspective of prediction time)
    eia_now = df["eia_forecast_mw"]  # EIA forecast for current hour
    actual_now = df["lag_24h"]  # Actual demand 24h ago (at prediction time)
    
    # ── EIA forecast errors (lags) ───────────────────────────────────────────
    # Error = EIA forecast - actual demand
    # These are known at training time (we have both values)
    # At prediction time, we compute error for t-48, t-72, etc.
    
    # Shift to get 48h-old EIA forecast error
    eia_48h = eia_now.shift(24)  # EIA forecast made 48h ago
    actual_48h = actual_now.shift(24)  # Actual demand 48h ago
    
    # EIA error 48h ago (known at prediction time)
    df["eia_error_48h"] = (eia_48h - actual_48h).astype("float32")
    
    # EIA error 48-72h ago (recent bias trend)
    eia_72h = eia_now.shift(48)
    actual_72h = actual_now.shift(48)
    df["eia_error_72h"] = (eia_72h - actual_72h).astype("float32")
    
    # ── EIA bias features ────────────────────────────────────────────────────
    # Rolling average of EIA errors = systematic bias
    df["eia_bias_24h"] = df["eia_error_48h"].rolling(24, min_periods=12).mean().astype("float32")
    df["eia_bias_48h"] = df["eia_error_48h"].rolling(48, min_periods=24).mean().astype("float32")
    
    # EIA bias by hour of day (systematic hourly patterns)
    if "hour_of_day" in df.columns:
        # Group by hour and compute rolling mean error for that hour
        df["eia_bias_by_hour"] = df.groupby("hour_of_day")["eia_error_48h"].transform(
            lambda x: x.rolling(168, min_periods=24).mean()  # 1 week of same hour
        ).astype("float32")
    
    # ── EIA forecast changes (momentum) ─────────────────────────────────────
    # How much EIA revised their forecast
    df["eia_forecast_change_24h"] = (eia_now - eia_48h).astype("float32")
    df["eia_forecast_change_48h"] = (eia_48h - eia_72h).astype("float32")
    
    # EIA forecast acceleration
    df["eia_forecast_accel"] = (
        df["eia_forecast_change_24h"] - df["eia_forecast_change_48h"]
    ).astype("float32")
    
    # ── EIA vs persistence ──────────────────────────────────────────────────
    # Does EIA beat naive persistence model?
    if "lag_48h" in df.columns:
        persistence = df["lag_48h"]  # Persistence: yesterday same time
        eia_error = (eia_now - actual_now).abs()
        persistence_error = (persistence - actual_now).abs()
        df["eia_vs_persistence"] = (eia_error < persistence_error).astype("int8")
    
    # ── EIA confidence indicators ────────────────────────────────────────────
    # High forecast variance = low confidence
    df["eia_forecast_volatility"] = eia_now.rolling(24, min_periods=12).std().astype("float32")
    
    # EIA forecast range (high - low over past week)
    df["eia_forecast_range_168h"] = (
        eia_now.rolling(168, min_periods=72).max() - 
        eia_now.rolling(168, min_periods=72).min()
    ).astype("float32")
    
    # ── EIA-weather interaction ─────────────────────────────────────────────
    # EIA performs worse in extreme weather
    if "temp_2m" in df.columns:
        temp = df["temp_2m"]
        # Hot day EIA bias
        df["eia_bias_hot"] = np.where(
            temp > 30, df["eia_bias_24h"], np.nan
        ).astype("float32")
        # Cold day EIA bias
        df["eia_bias_cold"] = np.where(
            temp < 0, df["eia_bias_24h"], np.nan
        ).astype("float32")
    
    # ── EIA error trend ─────────────────────────────────────────────────────
    # Is EIA getting better or worse recently?
    df["eia_error_trend"] = (
        df["eia_error_48h"].rolling(24, min_periods=12).mean() - 
        df["eia_error_48h"].rolling(168, min_periods=72).mean()
    ).astype("float32")
    
    return df
