"""
engineer.py — Feature matrix builder for energy load forecasting.

Reads from SQLite (populated by ingest pipeline) and writes per-BA
feature Parquet files to data/processed/features/.

Output schema (34 columns):
    demand_mw           float32   Target: actual hourly demand
    eia_forecast_mw     float32   EIA day-ahead forecast (benchmark + feature)
    hour_of_day         int8      0-23
    day_of_week         int8      0=Mon, 6=Sun
    month               int8      1-12
    is_weekend          int8      0/1
    is_us_holiday       int8      0/1  (US federal holidays)
    sin_hour_1/2        float32   Daily Fourier k=1,2
    cos_hour_1/2        float32   Daily Fourier k=1,2
    sin_week_1/2/3      float32   Weekly Fourier k=1,2,3
    cos_week_1/2/3      float32   Weekly Fourier k=1,2,3
    lag_1h/2h/24h/48h/168h  float32  Demand lags
    rolling_mean_24h    float32   24h rolling mean (closed='left', no leakage)
    rolling_std_24h     float32   24h rolling std
    rolling_mean_168h   float32   168h rolling mean
    rolling_std_168h    float32   168h rolling std
    wind_pct            float32   WND share of (WND+SUN+COL+NG) gen; 0 if no fuel data
    solar_pct           float32   SUN share
    coal_pct            float32   COL share
    ng_pct              float32   NG share
    respondent          category  BA code
    is_imputed          int8      From region_data (gap-filled row)
    is_anomaly          int8      From region_data (anomalous row)
    fold                int8      Walk-forward fold: year-2020, clipped to [-1, 5]

Walk-forward fold semantics:
    fold = -1  → 2019 rows (train-only anchor, never used for validation)
    fold =  0  → 2020 (validated in fold 0)
    fold =  k  → year 2020+k (validated in fold k, trained on folds < k)
    fold =  5  → 2025+ (holdout, NEVER touched during CV)
"""

from __future__ import annotations

import logging
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.data.database import get_engine
from src.data.ingest import load_wide

log = logging.getLogger(__name__)

# Fuel types tracked for mix features (share of these 4 fuels only)
_FUEL_TYPES = ["WND", "SUN", "COL", "NG"]
_FUEL_PCT_COLS = ["wind_pct", "solar_pct", "coal_pct", "ng_pct"]

# Warmup rows to drop (covers rolling_168h window)
_WARMUP_ROWS = 168

# US federal holidays for the full data range
_US_HOLIDAYS: set = set()


def _get_us_holidays(years: range) -> set:
    """Return a set of date objects for US federal holidays in the given years."""
    h = holidays.country_holidays("US", years=list(years))
    return set(h.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Individual feature group functions
# ─────────────────────────────────────────────────────────────────────────────


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour_of_day, day_of_week, month, is_weekend, is_us_holiday.

    Args:
        df: DataFrame with a UTC DatetimeIndex.

    Returns:
        df with new calendar columns appended (in-place modification).
    """
    idx = df.index
    df["hour_of_day"] = idx.hour.astype("int8")
    df["day_of_week"] = idx.dayofweek.astype("int8")
    df["month"] = idx.month.astype("int8")
    df["is_weekend"] = (idx.dayofweek >= 5).astype("int8")

    # Build holiday set for the full year range covered by this index
    years = range(idx.year.min(), idx.year.max() + 1)
    us_holidays = _get_us_holidays(years)

    # Normalize each timestamp to a date for O(1) lookup
    dates = idx.normalize().date  # array of date objects
    df["is_us_holiday"] = np.array([int(d in us_holidays) for d in dates], dtype="int8")

    return df


def add_fourier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily (k=1,2) and weekly (k=1,2,3) sin/cos Fourier terms.

    Args:
        df: DataFrame that already has hour_of_day and day_of_week columns.

    Returns:
        df with 10 new Fourier columns appended.
    """
    hour = df["hour_of_day"].to_numpy(dtype=float)
    dow = df["day_of_week"].to_numpy(dtype=float)
    hour_in_week = dow * 24.0 + hour

    for k in (1, 2):
        angle = 2.0 * np.pi * k * hour / 24.0
        df[f"sin_hour_{k}"] = np.sin(angle).astype("float32")
        df[f"cos_hour_{k}"] = np.cos(angle).astype("float32")

    for k in (1, 2, 3):
        angle = 2.0 * np.pi * k * hour_in_week / 168.0
        df[f"sin_week_{k}"] = np.sin(angle).astype("float32")
        df[f"cos_week_{k}"] = np.cos(angle).astype("float32")

    return df


def add_lag_features(
    df: pd.DataFrame,
    demand_col: str = "demand_mw",
) -> pd.DataFrame:
    """Add lag_1h, lag_2h, lag_24h, lag_48h, lag_168h from demand_col.

    Args:
        df: DataFrame with a strictly-hourly UTC DatetimeIndex.
        demand_col: Column to lag (default 'demand_mw').

    Returns:
        df with lag columns appended. First N rows will have NaN (handled by
        warmup drop downstream).
    """
    for lag in (1, 2, 24, 48, 168):
        df[f"lag_{lag}h"] = df[demand_col].shift(lag).astype("float32")
    return df


def add_rolling_features(
    df: pd.DataFrame,
    demand_col: str = "demand_mw",
) -> pd.DataFrame:
    """Add rolling_mean/std for 24h and 168h windows.

    Uses closed='left' so the current row is NOT included in its own window.
    This prevents data leakage at inference time.

    Args:
        df: DataFrame with demand_col.
        demand_col: Column to compute rolling stats on.

    Returns:
        df with 4 new rolling columns appended.
    """
    for w in (24, 168):
        roll = df[demand_col].rolling(window=w, min_periods=w // 2, closed="left")
        df[f"rolling_mean_{w}h"] = roll.mean().astype("float32")
        df[f"rolling_std_{w}h"] = roll.std().astype("float32")
    return df


def load_fuel_wide(engine, respondent: str) -> pd.DataFrame:
    """Load fuel_type_data for one BA as a wide hourly DataFrame.

    Returns DataFrame with columns [WND, SUN, COL, NG] indexed by period (UTC).
    Returns an empty DataFrame with those columns if no data is available.
    """
    q = text("""
        SELECT period, fueltype, value_mwh
        FROM fuel_type_data
        WHERE respondent = :r AND fueltype IN ('WND', 'SUN', 'COL', 'NG')
        ORDER BY period
    """)

    with engine.connect() as conn:
        raw = pd.read_sql_query(q, conn, params={"r": respondent}, parse_dates=["period"])

    if raw.empty:
        return pd.DataFrame(columns=_FUEL_TYPES)

    raw["period"] = pd.to_datetime(raw["period"], utc=True)
    wide = raw.pivot_table(
        index="period",
        columns="fueltype",
        values="value_mwh",
        aggfunc="first",
    )
    wide.columns.name = None

    # Ensure all 4 fuel columns exist even if a BA lacks certain fuel types
    for col in _FUEL_TYPES:
        if col not in wide.columns:
            wide[col] = 0.0

    return wide[_FUEL_TYPES]


def add_fuel_features(
    df: pd.DataFrame,
    engine,
    respondent: str,
) -> pd.DataFrame:
    """Add wind_pct, solar_pct, coal_pct, ng_pct from fuel_type_data table.

    Percentages represent each fuel's share of (WND + SUN + COL + NG) total
    generation per hour. If a BA has no fuel data, all pct columns are 0.0.

    Args:
        df: DataFrame with UTC DatetimeIndex.
        engine: SQLAlchemy engine.
        respondent: BA code.

    Returns:
        df with 4 fuel pct columns appended.
    """
    fuel = load_fuel_wide(engine, respondent)

    if fuel.empty:
        for col in _FUEL_PCT_COLS:
            df[col] = np.float32(0.0)
        return df

    # Align fuel index to demand index
    fuel = fuel.reindex(df.index)

    # Total generation (sum of 4 fuels only); replace 0 with NaN to avoid div-by-zero
    total = fuel[_FUEL_TYPES].sum(axis=1).replace(0.0, np.nan)

    for fuel_code, pct_col in zip(_FUEL_TYPES, _FUEL_PCT_COLS):
        pct = (fuel[fuel_code] / total).fillna(0.0)
        df[pct_col] = pct.values.astype("float32")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fold assignment
# ─────────────────────────────────────────────────────────────────────────────


def assign_fold(dt_index: pd.DatetimeIndex) -> pd.Series:
    """Assign walk-forward fold labels to a DatetimeIndex.

    fold = year - 2020, clipped to [-1, 5]:
        2019 → -1  (train-only anchor, never used for validation)
        2020 →  0  (first validation fold)
        2021 →  1
        2022 →  2
        2023 →  3
        2024 →  4
        2025+ →  5  (holdout — never touched during CV)

    Usage downstream:
        train_mask = df["fold"] < k   # expanding window
        val_mask   = df["fold"] == k

    Args:
        dt_index: UTC DatetimeIndex.

    Returns:
        int8 Series aligned to dt_index, name='fold'.
    """
    raw = np.array(dt_index.year, dtype="int16") - 2020
    clipped = np.clip(raw, -1, 5).astype("int8")
    return pd.Series(clipped, index=dt_index, name="fold")


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_flags(engine, respondent: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load is_imputed and is_anomaly flags for the D (demand) series.

    Returns DataFrame with columns [is_imputed, is_anomaly] aligned to index.
    Missing rows default to 0.
    """
    q = text("""
        SELECT period, is_imputed, is_anomaly
        FROM region_data
        WHERE respondent = :r AND type = 'D'
        ORDER BY period
    """)

    with engine.connect() as conn:
        flags = pd.read_sql_query(q, conn, params={"r": respondent}, parse_dates=["period"])

    if flags.empty:
        return pd.DataFrame(
            {"is_imputed": np.zeros(len(index), dtype="int8"),
             "is_anomaly": np.zeros(len(index), dtype="int8")},
            index=index,
        )

    flags["period"] = pd.to_datetime(flags["period"], utc=True)
    flags = flags.set_index("period")
    flags = flags.reindex(index).fillna(0)
    flags["is_imputed"] = flags["is_imputed"].astype("int8")
    flags["is_anomaly"] = flags["is_anomaly"].astype("int8")
    return flags


def _cast_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Cast all columns to their target dtypes per the output schema."""
    float32_cols = [
        "demand_mw", "eia_forecast_mw",
        "sin_hour_1", "cos_hour_1", "sin_hour_2", "cos_hour_2",
        "sin_week_1", "cos_week_1", "sin_week_2", "cos_week_2",
        "sin_week_3", "cos_week_3",
        "lag_1h", "lag_2h", "lag_24h", "lag_48h", "lag_168h",
        "rolling_mean_24h", "rolling_std_24h",
        "rolling_mean_168h", "rolling_std_168h",
        "wind_pct", "solar_pct", "coal_pct", "ng_pct",
    ]
    int8_cols = [
        "hour_of_day", "day_of_week", "month",
        "is_weekend", "is_us_holiday",
        "is_imputed", "is_anomaly", "fold",
    ]

    for col in float32_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    for col in int8_cols:
        if col in df.columns:
            df[col] = df[col].astype("int8")
    if "respondent" in df.columns:
        df["respondent"] = df["respondent"].astype("category")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main public API
# ─────────────────────────────────────────────────────────────────────────────


def build_features_for_ba(
    engine,
    respondent: str,
    output_dir: Path | None = None,
    save_parquet: bool = True,
) -> pd.DataFrame:
    """Build the complete feature matrix for one BA and optionally persist it.

    Args:
        engine:       SQLAlchemy engine pointing at the SQLite DB.
        respondent:   BA code (e.g. 'MISO', 'PJM').
        output_dir:   Directory to write {BA}_features.parquet.
                      Defaults to data/processed/features/ relative to project root.
        save_parquet: Whether to write the Parquet file.

    Returns:
        DataFrame with UTC DatetimeIndex, 34 feature columns, and demand_mw target.
        Returns empty DataFrame if BA has no D (demand) series.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "data" / "processed" / "features"

    # 1. Load D and DF (demand + EIA forecast)
    wide = load_wide(engine, respondent, types=["D", "DF"])

    if wide.empty or "D" not in wide.columns or wide["D"].isna().all():
        log.warning("%s: no D series found — skipping", respondent)
        return pd.DataFrame()

    df = wide.rename(columns={"D": "demand_mw", "DF": "eia_forecast_mw"})
    df["demand_mw"] = df["demand_mw"].astype("float32")
    df["eia_forecast_mw"] = df["eia_forecast_mw"].astype("float32")

    # 2. Flags passthrough
    flags = _load_flags(engine, respondent, df.index)
    df["is_imputed"] = flags["is_imputed"]
    df["is_anomaly"] = flags["is_anomaly"]

    # 3. Calendar + Fourier
    add_calendar_features(df)
    add_fourier_features(df)

    # 4. Lags + rolling (on demand_mw)
    add_lag_features(df)
    add_rolling_features(df)

    # 5. Fuel mix
    add_fuel_features(df, engine, respondent)

    # 6. Metadata
    df["respondent"] = respondent
    df["fold"] = assign_fold(df.index)

    # 7. Drop warmup rows and rows with no demand
    df = df.iloc[_WARMUP_ROWS:]
    df = df.dropna(subset=["demand_mw"])

    if df.empty:
        log.warning("%s: empty after warmup drop — skipping", respondent)
        return pd.DataFrame()

    # 8. Cast to schema dtypes
    df = _cast_schema(df)

    # 9. Persist
    if save_parquet:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{respondent}_features.parquet"
        df.to_parquet(path, compression="snappy", engine="pyarrow")

    return df


def build_features_all(
    engine,
    ba_codes: list[str],
    output_dir: Path | None = None,
    combined: bool = True,
) -> pd.DataFrame:
    """Build and persist feature matrices for a list of BAs.

    Args:
        engine:     SQLAlchemy engine.
        ba_codes:   List of BA codes to process.
        output_dir: Directory for output Parquet files.
        combined:   If True, stack all per-BA frames and save ALL_features.parquet.

    Returns:
        Combined DataFrame (all BAs stacked) if combined=True, else empty DataFrame.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "data" / "processed" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    total = len(ba_codes)

    for i, ba in enumerate(ba_codes, 1):
        df = build_features_for_ba(engine, ba, output_dir=output_dir, save_parquet=True)

        has_fuel = df["wind_pct"].any() if not df.empty else False
        fuel_tag = "fuel:yes" if has_fuel else "fuel:no"

        if df.empty:
            print(f"[{i:>3}/{total}]  {ba:<10} 0 rows  no D series — SKIPPED")
        else:
            path = output_dir / f"{ba}_features.parquet"
            print(
                f"[{i:>3}/{total}]  {ba:<10} {len(df):>7,} rows  "
                f"34 cols  {fuel_tag}  -> {path.name}"
            )
            frames.append(df)

    if combined and frames:
        all_df = pd.concat(frames, axis=0)
        combined_path = output_dir / "ALL_features.parquet"
        all_df.to_parquet(combined_path, compression="snappy", engine="pyarrow")
        print(f"\nCombined: {combined_path}  ({len(all_df):,} rows x {all_df.shape[1]} cols)")
        return all_df

    return pd.DataFrame()
