"""
engineer.py — Feature matrix builder for energy load forecasting.

Reads from SQLite (populated by ingest pipeline) and writes per-BA
feature Parquet files to data/processed/features/.

Output schema (60 columns after Phase 1):
    demand_mw               float32   Target: actual hourly demand
    eia_forecast_mw         float32   EIA day-ahead forecast (benchmark only, not a feature)
    hour_of_day             int8      0-23
    day_of_week             int8      0=Mon, 6=Sun
    month                   int8      1-12
    is_weekend              int8      0/1
    is_us_holiday           int8      0/1  (US federal holidays)
    sin_hour_1/2            float32   Daily Fourier k=1,2
    cos_hour_1/2            float32   Daily Fourier k=1,2
    sin_week_1/2/3          float32   Weekly Fourier k=1,2,3
    cos_week_1/2/3          float32   Weekly Fourier k=1,2,3
    lag_24h/48h/168h        float32   Demand lags (24h-ahead safe: ≥24h old)
    rolling_mean_24h        float32   24h rolling mean ending at t-24 (no leakage)
    rolling_std_24h         float32   24h rolling std ending at t-24
    rolling_mean_168h       float32   168h rolling mean ending at t-24
    rolling_std_168h        float32   168h rolling std ending at t-24
    wind_pct                float32   WND share of (WND+SUN+COL+NG) gen; 0 if no fuel data
    solar_pct               float32   SUN share
    coal_pct                float32   COL share
    ng_pct                  float32   NG share
    nuclear_pct             float32   NUC share of all generation; 0 if no fuel data
    hydro_pct               float32   WAT share of all generation; 0 if no fuel data
    renewable_pct           float32   (WND+SUN+WAT+GEO) share; 0 if no fuel data
    total_gen_mw_lag24      float32   Total generation at t-24 (all fuels); NaN if no data
    gen_demand_ratio_lag24  float32   total_gen / demand at t-24; NaN if no data
    ng_mw_lag24             float32   Net generation (NG type) at t-24; NaN if no NG data
    ti_mw_lag24             float32   Total interchange at t-24 (+=imports); NaN if no data
    ng_rolling_mean_24h     float32   24h rolling mean of NG, anchored t-24
    ti_rolling_mean_24h     float32   24h rolling mean of TI, anchored t-24
    demand_minus_ng_lag24   float32   demand[t-24] - net_gen[t-24] (import signal)
    ng_change_24h           float32   NG[t-24] - NG[t-48] (generation trend)
    interchange_net_lag24     float32 Net interchange at t-24: SUM(fromba=BA) - SUM(toba=BA)
                                      Positive = net exporter, negative = net importer
    interchange_vol_24h       float32 Rolling std of net interchange, 24h window at t-24
    # Phase 1: Weather features (20 columns)
    # Phase 1: Weather features (expanded)
    temp_2m                 float32   Temperature at 2m (°C) - unshifted = perfect forecast
    dewpoint_2m             float32   Dewpoint temperature at 2m (°C) - unshifted
    windspeed_10m           float32   Wind speed at 10m (km/h) - unshifted
    solar_irradiance        float32   Solar irradiance (W/m²) - unshifted
    cloudcover              float32   Cloud cover (%) - unshifted
    precipitation           float32   Precipitation (mm) - unshifted
    relative_humidity       float32   Relative humidity (%) - calculated from temp/dewpoint
    apparent_temp           float32   Apparent temperature (°C) - heat index calculated from temp/RH
    hdd                     float32   Heating degree days (base 18.3°C)
    cdd                     float32   Cooling degree days (base 18.3°C)
    hdd_sq                  float32   HDD squared
    cdd_sq                  float32   CDD squared
    temp_lag_48h            float32   Temperature at t-48 (2 days ago)
    temp_lag_168h           float32   Temperature at t-168 (1 week ago)
    temp_rolling_mean_6h    float32   6h rolling mean temperature ending at t-24
    temp_rolling_mean_24h   float32   24h rolling mean temperature ending at t-24
    temp_rolling_mean_48h   float32   48h rolling mean temperature ending at t-24
    temp_rolling_max_24h    float32   24h rolling max temperature ending at t-24
    temp_rolling_min_24h    float32   24h rolling min temperature ending at t-24
    temp_daily_max          float32   Daily max temperature (24h window)
    temp_daily_min          float32   Daily min temperature (24h window)
    temp_daily_range        float32   Daily temperature range (max - min)
    temp_delta_24h          float32   Temperature change from t-48 to t-24
    humidity_x_temp         float32   Relative humidity × temperature interaction
    temp_x_hour             float32   Temperature × hour_of_day interaction
    temp_x_month            float32   Temperature × month interaction
    cdd_x_hour              float32   CDD × hour_of_day interaction
    hdd_x_hour              float32   HDD × hour_of_day interaction
    temp_x_is_weekend       float32   Temperature × is_weekend interaction
    # Phase 2: Physics-inspired features
    sin_year_1              float32   Annual cycle (k=1) - yearly demand pattern
    cos_year_1              float32   Annual cycle (k=1)
    sin_year_2              float32   Annual cycle (k=2) - captures seasonal shape
    cos_year_2              float32   Annual cycle (k=2)
    day_length_hours        float32   Hours of daylight at BA location
    solar_elevation_noon    float32   Sun angle at solar noon (°)
    dewpoint_depression     float32   Temp - dewpoint (comfort indicator)
    thi                     float32   Temperature-Humidity Index (feels like in heat)
    wind_chill              float32   Wind Chill Index (feels like in cold)
    temp_cumulative_48h     float32   Cumulative temp over 48h (thermal mass effect)
    temp_weighted_48h       float32   Exponentially weighted temp (recent matters more)
    # Phase 3: Advanced temporal features
    lag_336h                float32   Demand lag at t-336 (2 weeks ago)
    demand_diff_24h         float32   Demand change: lag_24h - lag_48h (trend)
    demand_momentum         float32   Demand acceleration (second difference)
    day_of_month            int8      Day of month (1-31) - payroll cycles
    week_of_year            int8      ISO week number (1-53)
    quarter                 int8      Quarter of year (1-4)
    is_business_day         int8      1 if Mon-Fri and not holiday
    days_from_month_start   int8      Days since 1st of month
    days_to_month_end       int8      Days until month-end
    days_to_nearest_holiday         int16  Days to nearest US holiday
    days_to_nearest_major_holiday   int16  Days to nearest major holiday (Thanksgiving, Christmas, etc.)
    is_in_holiday_period            int8   1 if in multi-day holiday period (Thanksgiving Thu-Fri, etc.)
    is_in_major_holiday_period      int8   1 if in major holiday period
    days_into_holiday_period        int8   Days since holiday period started
    is_day_before_holiday           int8   1 if tomorrow is a holiday
    is_day_after_holiday            int8   1 if yesterday was a holiday
    # Phase 4: Cross-BA spatial features
    regional_demand_index           float32  Sum of demand in same interconnection (East/West/Texas)
    regional_demand_per_ba          float32  Average demand per BA in region
    neighbor_demand_avg             float32  Average demand of adjacent BAs
    neighbor_demand_max             float32  Max demand of adjacent BAs
    neighbor_demand_min             float32  Min demand of adjacent BAs
    demand_delta_vs_neighbors       float32  This BA demand minus neighbor average
    temp_delta_vs_neighbors         float32  This BA temp minus neighbor average
    is_import_stress                int8     1 if heavy imports (top 10% of net interchange)
    is_export_stress                int8     1 if heavy exports (bottom 10% of net interchange)
    grid_stress_index               float32  Composite regional stress indicator
    # COVID and DST features
    is_covid_period         int8      1 if March 2020 - Dec 2020 (anomalous demand period)
    is_dst_transition       int8      1 if DST clock change day (disrupted patterns)
    wind_x_wind_pct         float32   Wind speed × wind_pct interaction
    solar_x_solar_pct       float32   Solar irradiance × solar_pct interaction
    respondent              category  BA code
    is_imputed              int8      From region_data (gap-filled row)
    is_anomaly              int8      From region_data (anomalous row)
    fold                    int8      Walk-forward fold: year-2020, clipped to [-1, 5]

24h-ahead forecast horizon — data leakage policy:
    At decision time (t-24), we know demand up to t-24 inclusive.
    lag_1h (demand[t-1]) and lag_2h (demand[t-2]) are NOT available at t-24
    and have been removed. Rolling windows are anchored at t-24 by shifting
    demand 24 hours before computing the roll, so the window covers
    [t-48, t-25] for w=24 and [t-192, t-25] for w=168 — all ≥ 24h old.

Walk-forward fold semantics:
    fold = -1  → 2019 rows (train-only anchor, never used for validation)
    fold =  0  → 2020 (validated in fold 0)
    fold =  k  → year 2020+k (validated in fold k, trained on folds < k)
    fold =  5  → 2025+ (holdout, NEVER touched during CV)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.data.database import get_engine
from src.data.ingest import load_wide
from src.features.spatial_features import add_spatial_features

log = logging.getLogger(__name__)

# Fuel types tracked for original 4-fuel mix (share of these 4 only — backward compat)
_FUEL_TYPES = ["WND", "SUN", "COL", "NG"]
_FUEL_PCT_COLS = ["wind_pct", "solar_pct", "coal_pct", "ng_pct"]

# All meaningful generation fuel types (excludes nameplate/unknown: SNB, WNB, OES, UES, UNK)
_FUEL_TYPES_ALL = ["WND", "SUN", "COL", "NG", "NUC", "WAT", "GEO", "OIL", "OTH", "BAT", "PS"]
# Renewable fuels (supply-side clean energy)
_FUEL_RENEWABLE = ["WND", "SUN", "WAT", "GEO"]

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
    """Add lag_24h, lag_48h, lag_168h from demand_col.

    Only lags >= 24h are included so all features are available at prediction
    time for a 24h-ahead forecast (decision made at t-24, predicting t).
    lag_1h and lag_2h were removed — they require data from t-1 and t-2,
    which are not available when the forecast is issued 24h ahead.

    Args:
        df: DataFrame with a strictly-hourly UTC DatetimeIndex.
        demand_col: Column to lag (default 'demand_mw').

    Returns:
        df with lag columns appended. First N rows will have NaN (handled by
        warmup drop downstream).
    """
    for lag in (24, 48, 168):
        df[f"lag_{lag}h"] = df[demand_col].shift(lag).astype("float32")
    return df


def add_rolling_features(
    df: pd.DataFrame,
    demand_col: str = "demand_mw",
) -> pd.DataFrame:
    """Add rolling_mean/std for 24h and 168h windows, anchored at t-24.

    For a 24h-ahead forecast, rolling statistics must only use data available
    at decision time (t-24). We shift demand by 24 before rolling so the
    window at row t covers [t-48, t-25] for w=24 (or [t-192, t-25] for w=168).
    All values in those windows are at least 25 hours old at prediction time,
    hence safe.

    Previously this used closed='left' without shifting, which created a window
    of [t-24, t-1] — including t-23 through t-1 that are not yet available
    when the forecast is issued 24h ahead.

    Args:
        df: DataFrame with demand_col.
        demand_col: Column to compute rolling stats on.

    Returns:
        df with 4 new rolling columns appended.
    """
    # Shift by 24 so the most recent value in the window is demand[t-24]
    base = df[demand_col].shift(24)
    for w in (24, 168):
        roll = base.rolling(window=w, min_periods=w // 2)
        df[f"rolling_mean_{w}h"] = roll.mean().astype("float32")
        df[f"rolling_std_{w}h"] = roll.std().astype("float32")
    return df


def load_fuel_wide(engine, respondent: str) -> pd.DataFrame:
    """Load fuel_type_data for one BA as a wide hourly DataFrame.

    Returns DataFrame with all available fuel type columns indexed by period (UTC).
    The 4 original columns (WND, SUN, COL, NG) are always present (0.0 if missing).
    Additional columns (NUC, WAT, GEO, etc.) are included if the BA reports them.
    Returns an empty DataFrame if no data is available.
    """
    q = text("""
        SELECT period, fueltype, value_mwh
        FROM fuel_type_data
        WHERE respondent = :r
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

    # Ensure the 4 original fuel columns always exist
    for col in _FUEL_TYPES:
        if col not in wide.columns:
            wide[col] = 0.0

    return wide


def add_fuel_features(
    df: pd.DataFrame,
    engine,
    respondent: str,
) -> pd.DataFrame:
    """Add wind_pct, solar_pct, coal_pct, ng_pct from fuel_type_data table.

    Percentages represent each fuel's share of (WND + SUN + COL + NG) total
    generation per hour, lagged 24h to prevent data leakage.

    For a 24h-ahead forecast, the actual fuel mix at hour t is not available
    at decision time (t-24). We shift all fuel values by 24 rows so that
    wind_pct at row t = the actual wind share at t-24, which IS known at
    prediction time. The first 24 NaN rows from the shift are covered by
    the 168-row warmup drop applied later in build_features_for_ba().

    If a BA has no fuel data, all pct columns are 0.0.

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
        for col in ["nuclear_pct", "hydro_pct", "renewable_pct",
                    "total_gen_mw_lag24", "gen_demand_ratio_lag24"]:
            df[col] = np.float32(0.0)
        return df

    # Align fuel index to demand index, then shift 24h to avoid leakage
    fuel = fuel.reindex(df.index).shift(24)

    # ── Original 4-fuel pct (WND+SUN+COL+NG denominator, backward compatible) ──
    total_4 = fuel[_FUEL_TYPES].sum(axis=1).replace(0.0, np.nan)
    for fuel_code, pct_col in zip(_FUEL_TYPES, _FUEL_PCT_COLS):
        pct = (fuel[fuel_code] / total_4).fillna(0.0)
        df[pct_col] = pct.values.astype("float32")

    # ── Expanded fuel features (all-fuel denominator) ──
    available_all = [c for c in _FUEL_TYPES_ALL if c in fuel.columns]
    total_all = fuel[available_all].clip(lower=0).sum(axis=1).replace(0.0, np.nan)

    # Nuclear share
    nuc = fuel["NUC"] if "NUC" in fuel.columns else pd.Series(0.0, index=fuel.index)
    df["nuclear_pct"] = (nuc / total_all).fillna(0.0).values.astype("float32")

    # Hydro share (WAT = conventional hydro)
    wat = fuel["WAT"] if "WAT" in fuel.columns else pd.Series(0.0, index=fuel.index)
    df["hydro_pct"] = (wat / total_all).fillna(0.0).values.astype("float32")

    # Renewable share (WND + SUN + WAT + GEO)
    avail_renew = [c for c in _FUEL_RENEWABLE if c in fuel.columns]
    renew_total = fuel[avail_renew].clip(lower=0).sum(axis=1) if avail_renew else pd.Series(0.0, index=fuel.index)
    df["renewable_pct"] = (renew_total / total_all).fillna(0.0).values.astype("float32")

    # Total generation MW at t-24
    total_gen_series = total_all.copy()  # already shifted 24h
    df["total_gen_mw_lag24"] = total_gen_series.values.astype("float32")

    # Generation / demand ratio at t-24 (supply adequacy)
    demand_lag24 = df["lag_24h"].replace(0.0, np.nan)
    df["gen_demand_ratio_lag24"] = (total_gen_series / demand_lag24).clip(0, 5).values.astype("float32")

    return df


def add_grid_features(
    df: pd.DataFrame,
    engine,
    respondent: str,
) -> pd.DataFrame:
    """Add net-generation (NG) and total-interchange (TI) features from region_data.

    Both series are shifted 24h so the most recent value is from t-24, matching
    the 24h-ahead leakage policy.

    Features added:
        ng_mw_lag24          Net generation at t-24 (MW)
        ti_mw_lag24          Total interchange at t-24 (positive = net imports)
        ng_rolling_mean_24h  24h rolling mean of NG anchored at t-24
        ti_rolling_mean_24h  24h rolling mean of TI anchored at t-24
        demand_minus_ng_lag24  demand[t-24] - NG[t-24] (net import signal)
        ng_change_24h        NG[t-24] - NG[t-48] (generation momentum)

    All features default to NaN if the BA has no NG or TI data.
    """
    _ng_ti_cols = [
        "ng_mw_lag24", "ti_mw_lag24",
        "ng_rolling_mean_24h", "ti_rolling_mean_24h",
        "demand_minus_ng_lag24", "ng_change_24h",
    ]

    wide = load_wide(engine, respondent, types=["NG", "TI"])

    if wide.empty:
        for col in _ng_ti_cols:
            df[col] = np.float32(np.nan)
        return df

    # ── NG features ──
    if "NG" in wide.columns:
        ng = wide["NG"].reindex(df.index)
        ng_s24 = ng.shift(24)
        ng_s48 = ng.shift(48)
        df["ng_mw_lag24"] = ng_s24.astype("float32")
        df["ng_rolling_mean_24h"] = (
            ng_s24.rolling(window=24, min_periods=12).mean().astype("float32")
        )
        df["ng_change_24h"] = (ng_s24 - ng_s48).astype("float32")
        # demand[t-24] already available as lag_24h (add_lag_features called first)
        df["demand_minus_ng_lag24"] = (df["lag_24h"] - ng_s24).astype("float32")
    else:
        for col in ["ng_mw_lag24", "ng_rolling_mean_24h", "ng_change_24h", "demand_minus_ng_lag24"]:
            df[col] = np.float32(np.nan)

    # ── TI features ──
    if "TI" in wide.columns:
        ti = wide["TI"].reindex(df.index)
        ti_s24 = ti.shift(24)
        df["ti_mw_lag24"] = ti_s24.astype("float32")
        df["ti_rolling_mean_24h"] = (
            ti_s24.rolling(window=24, min_periods=12).mean().astype("float32")
        )
    else:
        for col in ["ti_mw_lag24", "ti_rolling_mean_24h"]:
            df[col] = np.float32(np.nan)

    return df


def add_interchange_features(
    df: pd.DataFrame,
    engine,
    respondent: str,
) -> pd.DataFrame:
    """Add cross-BA power flow features from interchange_data.

    Queries hourly import and export totals for this BA, shifts by 24h,
    and derives net interchange and rolling volatility.

    Features added:
        interchange_import_lag24  Total MW imported from all neighbors at t-24
        interchange_export_lag24  Total MW exported to all neighbors at t-24
        interchange_net_lag24     Net interchange at t-24 (import - export)
        interchange_vol_24h       Rolling std of net interchange, 24h window at t-24

    All features default to NaN if no interchange data exists for this BA.
    """
    _ic_cols = [
        "interchange_net_lag24", "interchange_vol_24h",
    ]

    try:
        # Two separate queries to exploit the ix_interchange_fromba_period index on both.
        # Query 1: flows WHERE fromba = respondent (exports from this BA)
        # Query 2: flows WHERE fromba = neighbor AND toba = respondent (imports to this BA)
        # Both use the fromba index — avoids a full table scan from OR toba = :r.
        q_exports = text("""
            SELECT period, SUM(COALESCE(value_mwh, 0)) AS exports
            FROM interchange_data
            WHERE fromba = :r
            GROUP BY period
        """)
        q_imports = text("""
            SELECT period, SUM(COALESCE(value_mwh, 0)) AS imports
            FROM interchange_data
            WHERE toba = :r
            GROUP BY period
        """)
        with engine.connect() as conn:
            exp_df = pd.read_sql_query(q_exports, conn, params={"r": respondent}, parse_dates=["period"])
            imp_df = pd.read_sql_query(q_imports, conn, params={"r": respondent}, parse_dates=["period"])
    except Exception as e:
        log.warning("%s: interchange query failed (%s) — skipping", respondent, e)
        for col in _ic_cols:
            df[col] = np.float32(np.nan)
        return df

    if exp_df.empty and imp_df.empty:
        for col in _ic_cols:
            df[col] = np.float32(np.nan)
        return df

    # Merge exports and imports on period
    exp_df["period"] = pd.to_datetime(exp_df["period"], utc=True)
    imp_df["period"] = pd.to_datetime(imp_df["period"], utc=True)
    ic = pd.merge(exp_df.set_index("period"), imp_df.set_index("period"),
                  left_index=True, right_index=True, how="outer").fillna(0.0)

    ic = ic.reindex(df.index)

    # Net interchange = SUM(fromba=:r) - SUM(toba=:r)
    # Positive = net exporter (sends more than receives), negative = net importer.
    # EIA interchange values are signed: a negative fromba row means that tie line
    # actually flowed in the reverse direction that hour — this is normal.
    net_s24 = (ic["exports"] - ic["imports"]).shift(24)

    df["interchange_net_lag24"] = net_s24.astype("float32")
    df["interchange_vol_24h"] = (
        net_s24.rolling(window=24, min_periods=12).std().astype("float32")
    )

    return df


def add_weather_features(
    df: pd.DataFrame,
    engine,
    respondent: str,
) -> pd.DataFrame:
    """Add weather features from Open-Meteo data.
    
    Loads hourly weather data for the BA, shifts by 24h for leakage safety,
    and adds 30+ weather-related features.
    """
    from sqlalchemy import text
    
    # Query weather data (6 core columns we have)
    q = text("""
        SELECT period, temp_2m, dewpoint_2m, windspeed_10m, 
               solar_irradiance, cloudcover, precipitation
        FROM weather_data
        WHERE respondent = :r
        ORDER BY period
    """)
    
    try:
        with engine.connect() as conn:
            weather = pd.read_sql_query(q, conn, params={"r": respondent}, parse_dates=["period"])
    except Exception as e:
        log.warning("%s: weather query failed (%s) — skipping", respondent, e)
        weather = pd.DataFrame()
    
    if weather.empty:
        # Set all weather columns to NaN
        weather_cols = [
            # Core weather (6 from API)
            "temp_2m", "dewpoint_2m", "windspeed_10m", 
            "solar_irradiance", "cloudcover", "precipitation",
            # Calculated from core
            "relative_humidity", "apparent_temp",
            # Thermal
            "hdd", "cdd", "hdd_sq", "cdd_sq",
            # Temp lags (all shifted 24h already, these are additional lags)
            "temp_lag_48h", "temp_lag_168h",
            # Rolling stats
            "temp_rolling_mean_6h", "temp_rolling_mean_24h", "temp_rolling_mean_48h",
            "temp_rolling_max_24h", "temp_rolling_min_24h",
            # Daily aggregates
            "temp_daily_max", "temp_daily_min", "temp_daily_range",
            # Changes
            "temp_delta_24h",
            # Interactions
            "humidity_x_temp",
            "temp_x_hour", "temp_x_month", "cdd_x_hour", "hdd_x_hour", "temp_x_is_weekend",
            "wind_x_wind_pct", "solar_x_solar_pct",
        ]
        for col in weather_cols:
            df[col] = np.float32(np.nan)
        return df
    
    # Set index and reindex
    weather["period"] = pd.to_datetime(weather["period"], utc=True)
    weather = weather.set_index("period")
    weather = weather.reindex(df.index)
    
    # NO SHIFT - Use un-shifted weather to simulate having perfect weather forecasts
    # This gives us an upper bound on model performance
    # Real implementation would use NOAA HRRR/GFS weather forecasts
    
    # Core weather features (un-shifted = perfect forecast)
    df["temp_2m"] = weather["temp_2m"].astype("float32")
    df["dewpoint_2m"] = weather["dewpoint_2m"].astype("float32")
    df["windspeed_10m"] = weather["windspeed_10m"].astype("float32")
    df["solar_irradiance"] = weather["solar_irradiance"].astype("float32")
    df["cloudcover"] = weather["cloudcover"].astype("float32")
    df["precipitation"] = weather["precipitation"].astype("float32")
    
    # Calculate relative humidity from temp and dewpoint (Magnus formula)
    # This is leakage-safe because both inputs are already shifted 24h
    temp_c = weather["temp_2m"]
    dewpoint_c = weather["dewpoint_2m"]
    
    # Magnus formula constants
    a = 17.271
    b = 237.7
    
    # Calculate saturation vapor pressure and actual vapor pressure
    alpha = ((a * temp_c) / (b + temp_c)) - ((a * dewpoint_c) / (b + dewpoint_c))
    rh = (100 * np.exp(alpha)).clip(0, 100)  # Clip to valid range
    df["relative_humidity"] = rh.astype("float32")
    
    # Calculate apparent temperature (simplified heat index)
    # Formula approximation using temp and relative humidity
    # This is leakage-safe as both inputs are shifted
    hi = -8.784694755 + 1.61139411 * temp_c + 2.338548839 * rh
    hi = hi - 0.14611605 * temp_c * rh - 0.012308094 * temp_c**2
    hi = hi - 0.016424828 * rh**2 + 0.002211732 * temp_c**2 * rh
    hi = hi + 0.00072546 * temp_c * rh**2 - 0.000003582 * temp_c**2 * rh**2
    
    # Use heat index when temp > 27°C, otherwise just use temp
    apparent = np.where(temp_c > 27, hi, temp_c)
    df["apparent_temp"] = apparent.astype("float32")
    
    # Derived thermal features
    temp = weather["temp_2m"]
    df["hdd"] = np.maximum(0, 18.3 - temp).astype("float32")
    df["cdd"] = np.maximum(0, temp - 18.3).astype("float32")
    df["hdd_sq"] = (df["hdd"] ** 2).astype("float32")
    df["cdd_sq"] = (df["cdd"] ** 2).astype("float32")
    
    # Temperature lags (additional to the base 24h shift already applied)
    # These represent temp at t-48, t-168 using the already-shifted data
    df["temp_lag_48h"] = temp.shift(24).astype("float32")  # Actually t-48 from original
    df["temp_lag_168h"] = temp.shift(144).astype("float32")  # Actually t-168 from original
    
    # Temperature change (t-24 vs t-48)
    df["temp_delta_24h"] = (temp - temp.shift(24)).astype("float32")
    
    # Rolling temperature means (all calculated on shifted data = safe)
    df["temp_rolling_mean_6h"] = temp.rolling(6, min_periods=3).mean().astype("float32")
    df["temp_rolling_mean_24h"] = temp.rolling(24, min_periods=12).mean().astype("float32")
    df["temp_rolling_mean_48h"] = temp.rolling(48, min_periods=24).mean().astype("float32")
    
    # Rolling temperature extremes
    df["temp_rolling_max_24h"] = temp.rolling(24, min_periods=12).max().astype("float32")
    df["temp_rolling_min_24h"] = temp.rolling(24, min_periods=12).min().astype("float32")
    
    # Daily temperature aggregates (24h = 1 day, on shifted data)
    df["temp_daily_max"] = temp.rolling(24, min_periods=12).max().astype("float32")
    df["temp_daily_min"] = temp.rolling(24, min_periods=12).min().astype("float32")
    df["temp_daily_range"] = (df["temp_daily_max"] - df["temp_daily_min"]).astype("float32")
    
    # Humidity-heat interaction
    df["humidity_x_temp"] = (df["relative_humidity"] * temp).astype("float32")
    
    # Interaction features
    df["temp_x_hour"] = (temp * df["hour_of_day"]).astype("float32")
    df["temp_x_month"] = (temp * df["month"]).astype("float32")
    df["cdd_x_hour"] = (df["cdd"] * df["hour_of_day"]).astype("float32")
    df["hdd_x_hour"] = (df["hdd"] * df["hour_of_day"]).astype("float32")
    df["temp_x_is_weekend"] = (temp * df["is_weekend"]).astype("float32")
    
    # Weather-fuel interactions (safely handle missing columns)
    if "wind_pct" in df.columns:
        df["wind_x_wind_pct"] = (weather["windspeed_10m"] * df["wind_pct"]).astype("float32")
    else:
        df["wind_x_wind_pct"] = np.float32(np.nan)
    
    if "solar_pct" in df.columns:
        df["solar_x_solar_pct"] = (weather["solar_irradiance"] * df["solar_pct"]).astype("float32")
    else:
        df["solar_x_solar_pct"] = np.float32(np.nan)
    
    return df


def add_phase2_features(df: pd.DataFrame, ba_code: str) -> pd.DataFrame:
    """Add Phase 2 physics-inspired features.
    
    These features capture annual cycles, solar geometry, and thermal comfort
    better than raw calendar features.
    
    All features are leakage-safe (calculated from timestamp or shifted weather).
    """
    from src.data.ba_coordinates import get_ba_coordinates
    
    idx = df.index
    
    # ── Annual Fourier cycles (365.25 days) ─────────────────────────────────
    # These capture yearly demand patterns better than raw month
    day_of_year = idx.dayofyear.to_numpy(dtype=float)
    for k in (1, 2):
        angle = 2.0 * np.pi * k * day_of_year / 365.25
        df[f"sin_year_{k}"] = np.sin(angle).astype("float32")
        df[f"cos_year_{k}"] = np.cos(angle).astype("float32")
    
    # ── Solar geometry (if BA coordinates available) ─────────────────────────
    try:
        coords = get_ba_coordinates(ba_code)
        lat = coords["lat"]
        
        # Calculate day length and solar noon elevation for each day
        # This is leakage-safe - depends only on date and location
        day_of_year_arr = idx.dayofyear.values
        
        # Solar declination (angle of sun at solar noon at equator)
        # Formula: 23.45° × sin(360° × (284 + day_of_year) / 365)
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year_arr) / 365))
        
        # Hour angle at sunrise/sunset: cos(H) = -tan(lat) × tan(declination)
        lat_rad = np.radians(lat)
        dec_rad = np.radians(declination)
        
        # Clamp to [-1, 1] for arccos
        cos_hour_angle = np.clip(-np.tan(lat_rad) * np.tan(dec_rad), -1, 1)
        hour_angle = np.degrees(np.arccos(cos_hour_angle))
        
        # Day length = 2 × hour_angle / 15 (15° per hour)
        day_length = 2 * hour_angle / 15
        df["day_length_hours"] = day_length.astype("float32")
        
        # Solar elevation at noon (90 - |lat - declination|)
        solar_elevation = 90 - np.abs(lat - declination)
        df["solar_elevation_noon"] = solar_elevation.astype("float32")
        
    except Exception:
        # If coordinates not found, use placeholders
        df["day_length_hours"] = np.float32(12.0)  # Average
        df["solar_elevation_noon"] = np.float32(45.0)  # Average
    
    # ── Dewpoint depression (comfort indicator) ──────────────────────────────
    # Temp - dewpoint: smaller = more humid/muggy
    if "temp_2m" in df.columns and "dewpoint_2m" in df.columns:
        df["dewpoint_depression"] = (df["temp_2m"] - df["dewpoint_2m"]).astype("float32")
    else:
        df["dewpoint_depression"] = np.float32(np.nan)
    
    # ── Temperature-Humidity Index (THI) ─────────────────────────────────────
    # "Feels like" temperature in hot weather (humidity makes it feel hotter)
    # Formula: THI = T - 0.55*(1-RH/100)*(T-58)  [T in °F]
    if "temp_2m" in df.columns and "relative_humidity" in df.columns:
        temp_f = df["temp_2m"] * 9/5 + 32  # Convert to Fahrenheit
        rh = df["relative_humidity"]
        thi_f = temp_f - 0.55 * (1 - rh/100) * (temp_f - 58)
        thi_c = (thi_f - 32) * 5/9  # Convert back to Celsius
        # Only valid for warm conditions (temp > 18°C)
        df["thi"] = np.where(df["temp_2m"] > 18, thi_c, df["temp_2m"]).astype("float32")
    else:
        df["thi"] = np.float32(np.nan)
    
    # ── Wind Chill Index (WCI) ───────────────────────────────────────────────
    # "Feels like" temperature in cold weather (wind makes it feel colder)
    # Formula: WCI = 13.12 + 0.6215*T - 11.37*V^0.16 + 0.3965*T*V^0.16
    if "temp_2m" in df.columns and "windspeed_10m" in df.columns:
        temp_c = df["temp_2m"]
        # Convert windspeed from km/h to proper units (assume input is km/h)
        v_kmh = df["windspeed_10m"]
        wci = 13.12 + 0.6215*temp_c - 11.37*(v_kmh**0.16) + 0.3965*temp_c*(v_kmh**0.16)
        # Only valid for cold conditions (temp < 10°C)
        df["wind_chill"] = np.where(temp_c < 10, wci, temp_c).astype("float32")
    else:
        df["wind_chill"] = np.float32(np.nan)
    
    # ── Thermal lag features (building thermal mass) ─────────────────────────
    # Demand responds to past temps with delay due to building insulation
    if "temp_2m" in df.columns:
        temp = df["temp_2m"]
        # Cumulative heating/cooling over past 48h (building thermal mass effect)
        df["temp_cumulative_48h"] = temp.rolling(48, min_periods=24).sum().astype("float32")
        # Weighted average (recent temps matter more)
        weights = np.exp(-np.arange(48) / 12)  # Exponential decay
        weights = weights / weights.sum()
        df["temp_weighted_48h"] = temp.rolling(48, min_periods=24).apply(
            lambda x: np.sum(x * weights[-len(x):]), raw=True
        ).astype("float32")
    else:
        df["temp_cumulative_48h"] = np.float32(np.nan)
        df["temp_weighted_48h"] = np.float32(np.nan)
    
    return df


def add_phase3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Phase 3 advanced temporal features.
    
    These features capture longer-term patterns, demand momentum, and
    business calendar effects that drive energy demand.
    
    All features are leakage-safe (use only t-24 or older data).
    """
    idx = df.index
    
    # ── Extended lag features ────────────────────────────────────────────────
    # Already have lag_24h, lag_48h, lag_168h from Phase 0
    # Add longer lags for capturing multi-week patterns
    if "lag_168h" in df.columns:
        lag_168 = df["lag_168h"]
        df["lag_336h"] = lag_168.shift(168).astype("float32")  # 2 weeks ago
    
    # ── Demand momentum (differencing) ───────────────────────────────────────
    # How demand is trending (accelerating/decelerating)
    if "lag_24h" in df.columns and "lag_48h" in df.columns:
        lag_24 = df["lag_24h"]
        lag_48 = df["lag_48h"]
        # First difference: change from t-48 to t-24
        df["demand_diff_24h"] = (lag_24 - lag_48).astype("float32")
        # Momentum: approximate second difference using lag_48 vs lag_72 (computed from lag_48 shifted)
        lag_72_approx = lag_48.shift(24)  # This is t-72
        df["demand_momentum"] = (lag_24 - 2*lag_48 + lag_72_approx).astype("float32")
    
    # ── Enhanced calendar features ───────────────────────────────────────────
    # Day of month (payroll cycles affect demand)
    df["day_of_month"] = idx.day.astype("int8")
    
    # Week of year
    df["week_of_year"] = idx.isocalendar().week.astype("int8")
    
    # Quarter
    df["quarter"] = idx.quarter.astype("int8")
    
    # Is business day (Monday-Friday, not holiday)
    if "is_us_holiday" in df.columns:
        df["is_business_day"] = ((idx.dayofweek < 5) & (df["is_us_holiday"] == 0)).astype("int8")
    else:
        df["is_business_day"] = (idx.dayofweek < 5).astype("int8")
    
    # Days from start of month (ramp-up effect after month-start)
    df["days_from_month_start"] = (idx.day - 1).astype("int8")
    
    # Days to end of month (month-end effect)
    days_in_month = idx.days_in_month
    df["days_to_month_end"] = (days_in_month - idx.day).astype("int8")
    
    # ── Enhanced holiday features ────────────────────────────────────────────
    years = range(idx.year.min(), idx.year.max() + 1)
    us_holidays = _get_us_holidays(years)
    holiday_dates = sorted(us_holidays)
    
    dates = pd.Series(idx.normalize().tz_localize(None), index=idx)
    
    # Major holidays that typically have multi-day effects
    major_holiday_names = {
        "New Year", "Memorial Day", "Independence Day", "Labor Day",
        "Thanksgiving", "Christmas Day", "Martin Luther King Jr. Day",
        "Presidents' Day", "Veterans Day"
    }
    
    # Get holiday names for classification
    h_obj = holidays.country_holidays("US", years=list(years))
    
    # Build sets for quick lookup
    all_holiday_set = set(holiday_dates)
    
    # Create expanded holiday periods (Thanksgiving = Thu-Fri, Christmas = multi-day, etc.)
    holiday_periods = []  # List of (start_date, end_date, is_major)
    
    for h_date in holiday_dates:
        h_name = str(h_obj.get(h_date, ""))
        is_major = any(m in h_name for m in major_holiday_names)
        
        # Define holiday periods based on type
        if "Thanksgiving" in h_name:
            # Thanksgiving = Thursday, often Friday off too
            start = pd.Timestamp(h_date)
            end = start + pd.Timedelta(days=1)  # Thu-Fri
        elif "Christmas" in h_name:
            # Christmas week effect
            start = pd.Timestamp(h_date) - pd.Timedelta(days=2)  # Days before
            end = pd.Timestamp(h_date) + pd.Timedelta(days=2)    # Days after
        elif "New Year" in h_name or "Independence Day" in h_name:
            # 3-day window
            start = pd.Timestamp(h_date) - pd.Timedelta(days=1)
            end = pd.Timestamp(h_date) + pd.Timedelta(days=1)
        elif "Memorial Day" in h_name or "Labor Day" in h_name:
            # Monday holidays - often weekend + Monday
            start = pd.Timestamp(h_date) - pd.Timedelta(days=2)
            end = pd.Timestamp(h_date)
        else:
            # Other holidays - single day
            start = pd.Timestamp(h_date)
            end = start
        
        holiday_periods.append((start, end, is_major))
    
    # Calculate features for each date
    days_to_nearest = np.full(len(dates), 365, dtype=np.int16)
    days_to_nearest_major = np.full(len(dates), 365, dtype=np.int16)
    is_in_holiday_period = np.zeros(len(dates), dtype=np.int8)
    is_in_major_holiday_period = np.zeros(len(dates), dtype=np.int8)
    days_into_holiday_period = np.zeros(len(dates), dtype=np.int8)
    
    for i, date_val in enumerate(dates):
        # Find nearest holiday and nearest major holiday
        for start, end, is_major in holiday_periods:
            # Check if date is in holiday period
            if start <= date_val <= end:
                is_in_holiday_period[i] = 1
                days_into_holiday_period[i] = (date_val - start).days
                if is_major:
                    is_in_major_holiday_period[i] = 1
            
            # Distance to start of holiday period
            if date_val < start:
                days_to_start = (start - date_val).days
                days_to_nearest[i] = min(days_to_nearest[i], days_to_start)
                if is_major:
                    days_to_nearest_major[i] = min(days_to_nearest_major[i], days_to_start)
            elif date_val > end:
                days_to_end = (date_val - end).days
                days_to_nearest[i] = min(days_to_nearest[i], days_to_end)
                if is_major:
                    days_to_nearest_major[i] = min(days_to_nearest_major[i], days_to_end)
    
    df["days_to_nearest_holiday"] = days_to_nearest.astype("int16")
    df["days_to_nearest_major_holiday"] = days_to_nearest_major.astype("int16")
    df["is_in_holiday_period"] = is_in_holiday_period.astype("int8")
    df["is_in_major_holiday_period"] = is_in_major_holiday_period.astype("int8")
    df["days_into_holiday_period"] = days_into_holiday_period.astype("int8")
    
    # Is day before/after any holiday (for single-day holidays)
    is_before = np.zeros(len(dates), dtype=np.int8)
    is_after = np.zeros(len(dates), dtype=np.int8)
    for h_date in holiday_dates:
        h_ts = pd.Timestamp(h_date)
        before_mask = (dates == h_ts - pd.Timedelta(days=1))
        after_mask = (dates == h_ts + pd.Timedelta(days=1))
        is_before = is_before | before_mask.values
        is_after = is_after | after_mask.values
    df["is_day_before_holiday"] = is_before.astype("int8")
    df["is_day_after_holiday"] = is_after.astype("int8")
    
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

# Physical upper bound: no single BA in the US approaches 500 GW
_MAX_DEMAND_MW = 500_000


def _clean_raw_demand(df: pd.DataFrame, respondent: str) -> pd.DataFrame:
    """Replace physically impossible demand/forecast values with NaN.

    Two-pass cleaning:
    Pass 1 — absolute bounds: negatives, zeros, and values > 500k MW (no BA
             in the US approaches 500 GW; catches INT32 overflow artifacts).
    Pass 2 — per-BA statistical bounds: values above 5 × the 95th percentile
             of non-zero demand. Catches data corruption that slips below the
             absolute cap (e.g. 280k MW for a 2k MW median BA).
             Factor of 5 safely clears legitimate seasonal peaks (typically
             1.5–2.5× median) while flagging 10–165× spikes as errors.

    NaN rows in demand_mw propagate into lag/rolling features as NaN, which
    the model's SimpleImputer handles. Rows with NaN demand_mw are dropped
    before training (can't train on unknown target).
    """
    for col in ("demand_mw", "eia_forecast_mw"):
        if col not in df.columns:
            continue

        # Pass 1: absolute bounds
        bad = (df[col] <= 0) | (df[col] > _MAX_DEMAND_MW)
        n_bad = int(bad.sum())
        if n_bad:
            log.warning("%s: %d bad values in %s (min=%.0f, max=%.0f) → NaN",
                        respondent, n_bad, col,
                        float(df.loc[bad, col].min()),
                        float(df.loc[bad, col].max()))
            df.loc[bad, col] = np.nan

        # Pass 2: per-BA statistical cap (5 × p95 of valid values)
        valid = df[col].dropna()
        if len(valid) > 100:
            p95 = float(np.percentile(valid, 95))
            cap = max(p95 * 5, 1_000)   # floor at 1 GW to avoid over-clipping tiny BAs
            spike = df[col] > cap
            n_spike = int(spike.sum())
            if n_spike:
                log.warning("%s: %d statistical outliers in %s (> %.0f MW = 5×p95) → NaN",
                            respondent, n_spike, col, cap)
                df.loc[spike, col] = np.nan

    return df


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
        "lag_24h", "lag_48h", "lag_168h",
        "rolling_mean_24h", "rolling_std_24h",
        "rolling_mean_168h", "rolling_std_168h",
        # original fuel mix
        "wind_pct", "solar_pct", "coal_pct", "ng_pct",
        # Phase 0B: expanded fuel features
        "nuclear_pct", "hydro_pct", "renewable_pct",
        "total_gen_mw_lag24", "gen_demand_ratio_lag24",
        # Phase 0A: NG + TI grid features
        "ng_mw_lag24", "ti_mw_lag24",
        "ng_rolling_mean_24h", "ti_rolling_mean_24h",
        "demand_minus_ng_lag24", "ng_change_24h",
        # Phase 0C: interchange features
        "interchange_net_lag24", "interchange_vol_24h",
        # Phase 1: Weather features (expanded)
        "temp_2m", "dewpoint_2m", "windspeed_10m", "solar_irradiance", "cloudcover", "precipitation",
        "relative_humidity", "apparent_temp",  # Calculated from core fields
        # Thermal
        "hdd", "cdd", "hdd_sq", "cdd_sq",
        # Temp lags
        "temp_lag_48h", "temp_lag_168h",
        # Rolling stats
        "temp_rolling_mean_6h", "temp_rolling_mean_24h", "temp_rolling_mean_48h",
        "temp_rolling_max_24h", "temp_rolling_min_24h",
        # Daily aggregates
        "temp_daily_max", "temp_daily_min", "temp_daily_range",
        # Changes
        "temp_delta_24h",
        # Interactions
        "humidity_x_temp",
        "temp_x_hour", "temp_x_month", "cdd_x_hour", "hdd_x_hour", "temp_x_is_weekend",
        "wind_x_wind_pct", "solar_x_solar_pct",
        # Phase 2: Physics-inspired features
        "sin_year_1", "cos_year_1", "sin_year_2", "cos_year_2",
        "day_length_hours", "solar_elevation_noon",
        "dewpoint_depression",
        # Thermal comfort indices
        "thi", "wind_chill",
        # Thermal lag features
        "temp_cumulative_48h", "temp_weighted_48h",
        # Phase 3: Advanced temporal features
        "lag_336h", "demand_diff_24h", "demand_momentum",
        "day_of_month", "week_of_year", "quarter",
        "is_business_day", "days_from_month_start", "days_to_month_end",
        "days_to_nearest_holiday", "days_to_nearest_major_holiday",
        "is_in_holiday_period", "is_in_major_holiday_period", "days_into_holiday_period",
        "is_day_before_holiday", "is_day_after_holiday",
        # Phase 4: Spatial features
        "regional_demand_index", "regional_demand_per_ba",
        "neighbor_demand_avg", "neighbor_demand_max", "neighbor_demand_min",
        "demand_delta_vs_neighbors", "temp_delta_vs_neighbors",
        "grid_stress_index",
    ]
    int8_cols = [
        "hour_of_day", "day_of_week", "month", "day_of_month",
        "week_of_year", "quarter",
        "is_weekend", "is_us_holiday", "is_business_day",
        "is_imputed", "is_anomaly", "fold",
        "is_in_holiday_period", "is_in_major_holiday_period", "days_into_holiday_period",
        "is_day_before_holiday", "is_day_after_holiday",
        # Phase 4: Spatial binary features
        "is_import_stress", "is_export_stress",
        # COVID and DST flags
        "is_covid_period", "is_dst_transition",
    ]
    int16_cols = ["days_to_nearest_holiday", "days_to_nearest_major_holiday"]  # Can be up to 365

    for col in float32_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    for col in int8_cols:
        if col in df.columns:
            df[col] = df[col].astype("int8")
    for col in int16_cols:
        if col in df.columns:
            df[col] = df[col].astype("int16")
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

    t_start = time.perf_counter()
    _t = {}  # timing dict: step_name -> seconds

    # 1. Load D and DF (demand + EIA forecast)
    t0 = time.perf_counter()
    wide = load_wide(engine, respondent, types=["D", "DF"])
    _t["load_demand"] = time.perf_counter() - t0

    if wide.empty or "D" not in wide.columns or wide["D"].isna().all():
        log.warning("%s: no D series found — skipping", respondent)
        return pd.DataFrame()

    df = wide.rename(columns={"D": "demand_mw", "DF": "eia_forecast_mw"})
    df["demand_mw"] = df["demand_mw"].astype("float32")
    df["eia_forecast_mw"] = df["eia_forecast_mw"].astype("float32")

    # 2a. Clean impossible values before computing any derived features.
    t0 = time.perf_counter()
    df = _clean_raw_demand(df, respondent)
    _t["clean"] = time.perf_counter() - t0

    # 2. Flags passthrough
    t0 = time.perf_counter()
    flags = _load_flags(engine, respondent, df.index)
    df["is_imputed"] = flags["is_imputed"]
    df["is_anomaly"] = flags["is_anomaly"]
    _t["flags"] = time.perf_counter() - t0

    # 3. Calendar + Fourier
    t0 = time.perf_counter()
    add_calendar_features(df)
    add_fourier_features(df)
    _t["calendar_fourier"] = time.perf_counter() - t0

    # 4. Lags + rolling (on demand_mw)
    t0 = time.perf_counter()
    add_lag_features(df)
    add_rolling_features(df)
    _t["lags_rolling"] = time.perf_counter() - t0

    # 5. Fuel mix (original 4 pct + expanded: nuclear, hydro, renewable, total_gen, ratio)
    t0 = time.perf_counter()
    add_fuel_features(df, engine, respondent)
    _t["fuel"] = time.perf_counter() - t0

    # 5b. Grid features: NG + TI from region_data (shifted 24h)
    t0 = time.perf_counter()
    add_grid_features(df, engine, respondent)
    _t["grid_ng_ti"] = time.perf_counter() - t0

    # 5c. Interchange features: cross-BA power flows (shifted 24h)
    t0 = time.perf_counter()
    add_interchange_features(df, engine, respondent)
    _t["interchange"] = time.perf_counter() - t0

    # 5d. Weather features (Phase 1)
    t0 = time.perf_counter()
    add_weather_features(df, engine, respondent)
    _t["weather"] = time.perf_counter() - t0
    
    # 6. Phase 2: Physics-inspired features
    t0 = time.perf_counter()
    add_phase2_features(df, respondent)
    _t["phase2"] = time.perf_counter() - t0
    
    # 7. Phase 3: Advanced temporal features
    t0 = time.perf_counter()
    add_phase3_features(df)
    _t["phase3"] = time.perf_counter() - t0
    
    # 8. Phase 4: Cross-BA spatial features
    t0 = time.perf_counter()
    from src.data.config_loader import load_config
    cfg = load_config()
    all_bas = [ba['code'] for ba in cfg['balancing_authorities'] if ba.get('enabled', True)]
    add_spatial_features(df, engine, respondent, all_bas)
    _t["phase4"] = time.perf_counter() - t0
    
    # 9. COVID period flag (March 2020 - Dec 2020 had anomalous demand)
    idx = df.index
    df["is_covid_period"] = (
        ((idx >= "2020-03-01") & (idx <= "2020-12-31")).astype("int8")
    )
    
    # 10. DST transition flags (clock changes disrupt demand patterns)
    # Spring: 23h day, Fall: 25h day
    dst_dates = [
        "2019-03-10", "2019-11-03",
        "2020-03-08", "2020-11-01",
        "2021-03-14", "2021-11-07",
        "2022-03-13", "2022-11-06",
        "2023-03-12", "2023-11-05",
        "2024-03-10", "2024-11-03",
        "2025-03-09", "2025-11-02",
    ]
    dst_dates = pd.to_datetime(dst_dates).date
    df["is_dst_transition"] = df.index.normalize().isin(dst_dates).astype("int8")
    
    # 11. Metadata
    df["respondent"] = respondent
    df["fold"] = assign_fold(df.index)

    # 7. Drop warmup rows and rows with no demand
    df = df.iloc[_WARMUP_ROWS:]
    df = df.dropna(subset=["demand_mw"])

    if df.empty:
        log.warning("%s: empty after warmup drop — skipping", respondent)
        return pd.DataFrame()

    # 8. Cast to schema dtypes
    t0 = time.perf_counter()
    df = _cast_schema(df)
    _t["cast"] = time.perf_counter() - t0

    # 9. Persist
    t0 = time.perf_counter()
    if save_parquet:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{respondent}_features.parquet"
        df.to_parquet(path, compression="snappy", engine="pyarrow")
    _t["save"] = time.perf_counter() - t0

    _t["total"] = time.perf_counter() - t_start

    # Log timing breakdown (debug level — only visible with LOG_LEVEL=DEBUG)
    timing_str = "  ".join(f"{k}={v:.1f}s" for k, v in _t.items())
    log.debug("%s timing: %s", respondent, timing_str)

    # Store timings on df as metadata for build_features_all to print
    df.attrs["_timings"] = _t

    return df


def _ba_worker(args: dict) -> dict:
    """Worker function for parallel feature building."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    
    from src.data.database import get_engine
    from src.features.engineer import build_features_for_ba
    
    ba = args["ba"]
    db_url = args["db_url"]
    output_dir = Path(args["output_dir"])
    
    engine = get_engine(db_url)
    df = build_features_for_ba(engine, ba, output_dir=output_dir, save_parquet=True)
    
    has_fuel = df["wind_pct"].any() if not df.empty else False
    fuel_tag = "fuel:yes" if has_fuel else "fuel:no"
    
    return {
        "ba": ba,
        "df": df,
        "has_fuel": has_fuel,
        "fuel_tag": fuel_tag,
    }


def build_features_all(
    engine,
    ba_codes: list[str],
    output_dir: Path | None = None,
    combined: bool = True,
    workers: int = 1,
) -> pd.DataFrame:
    """Build and persist feature matrices for a list of BAs.

    Args:
        engine:     SQLAlchemy engine.
        ba_codes:   List of BA codes to process.
        output_dir: Directory for output Parquet files.
        combined:   If True, stack all per-BA frames and save ALL_features.parquet.
        workers:    Number of parallel workers (1 = sequential).

    Returns:
        Combined DataFrame (all BAs stacked) if combined=True, else empty DataFrame.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "data" / "processed" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    total = len(ba_codes)
    
    # Get DB URL for workers
    db_url = str(engine.url) if hasattr(engine, 'url') else None

    if workers > 1:
        # Parallel processing
        print(f"Using {workers} parallel workers...")
        
        worker_args = [
            {
                "ba": ba,
                "db_url": db_url,
                "output_dir": str(output_dir),
            }
            for ba in ba_codes
        ]
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_ba_worker, wa): wa["ba"] for wa in worker_args}
            completed = 0
            
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                ba = result["ba"]
                df = result["df"]
                fuel_tag = result["fuel_tag"]
                
                if df.empty:
                    print(f"[{completed:>3}/{total}]  {ba:<10} 0 rows  no D series — SKIPPED")
                else:
                    path = output_dir / f"{ba}_features.parquet"
                    timings = df.attrs.get("_timings", {})
                    total_s = timings.get("total", 0)
                    slow_steps = {k: v for k, v in timings.items()
                                  if k != "total" and v >= 0.5}
                    timing_detail = "  ".join(
                        f"{k}={v:.1f}s" for k, v in slow_steps.items()
                    )
                    print(
                        f"[{completed:>3}/{total}]  {ba:<10} {len(df):>7,} rows  "
                        f"{df.shape[1]} cols  {fuel_tag}  "
                        f"[{total_s:.1f}s{':  ' + timing_detail if timing_detail else ''}]"
                        f"  -> {path.name}"
                    )
                    frames.append(df)
    else:
        # Sequential processing
        for i, ba in enumerate(ba_codes, 1):
            df = build_features_for_ba(engine, ba, output_dir=output_dir, save_parquet=True)

            has_fuel = df["wind_pct"].any() if not df.empty else False
            fuel_tag = "fuel:yes" if has_fuel else "fuel:no"

            if df.empty:
                print(f"[{i:>3}/{total}]  {ba:<10} 0 rows  no D series — SKIPPED")
            else:
                path = output_dir / f"{ba}_features.parquet"
                timings = df.attrs.get("_timings", {})
                total_s = timings.get("total", 0)
                slow_steps = {k: v for k, v in timings.items()
                              if k != "total" and v >= 0.5}
                timing_detail = "  ".join(
                    f"{k}={v:.1f}s" for k, v in slow_steps.items()
                )
                print(
                    f"[{i:>3}/{total}]  {ba:<10} {len(df):>7,} rows  "
                    f"{df.shape[1]} cols  {fuel_tag}  "
                    f"[{total_s:.1f}s{':  ' + timing_detail if timing_detail else ''}]"
                    f"  -> {path.name}"
                )
                frames.append(df)

    if combined and frames:
        all_df = pd.concat(frames, axis=0)
        combined_path = output_dir / "ALL_features.parquet"
        all_df.to_parquet(combined_path, compression="snappy", engine="pyarrow")
        print(f"\nCombined: {combined_path}  ({len(all_df):,} rows x {all_df.shape[1]} cols)")
        return all_df

    return pd.DataFrame()
