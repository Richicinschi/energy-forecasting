"""
ingest.py — Data ingestion pipeline for EIA-930 raw CSV files.

Handles all 4 endpoint types:
  region-data      → region_data table      (D, DF, NG, TI per BA)
  fuel-type-data   → fuel_type_data table
  interchange-data → interchange_data table
  region-sub-ba-data → sub_ba_data table

For region-data only (the modelling table):
  - Validates schema and completeness
  - Reindexes to strict hourly frequency
  - Fills gaps ≤ MAX_GAP_HOURS via linear interpolation (marks is_imputed=1)
  - Flags anomalies: |value - rolling_mean_168h| > ANOMALY_STD * rolling_std_168h

Usage:
    from src.data.ingest import ingest_file, ingest_directory
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from src.data.database import (
    FuelTypeData,
    IngestLog,
    InterchangeData,
    RegionData,
    SubBaData,
    get_engine,
    create_all_tables,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MAX_GAP_HOURS = 6         # interpolate gaps up to this many consecutive NaNs
ANOMALY_STD = 4.0         # flag if |value - rolling_mean| > N * rolling_std
ROLLING_WINDOW = 168      # hours (1 week) for anomaly detection rolling stats
CHUNK_SIZE = 100          # rows per DB upsert batch (SQLite limit: 999 vars / 8 cols = 124 max)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def ingest_file(
    csv_path: str | Path,
    engine=None,
    db_url: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Ingest one raw EIA CSV file into the database.

    Auto-detects the endpoint type from the filename suffix.

    Args:
        csv_path: Path to a raw CSV (e.g. data/raw/region-data/MISO_*_region-data.csv)
        engine: SQLAlchemy engine (created from db_url or env if None)
        db_url: Database URL override
        verbose: Print progress

    Returns:
        Dict with keys: endpoint, respondent, rows_raw, rows_ingested,
                        rows_imputed, rows_anomaly, status, error_msg
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    if engine is None:
        engine = get_engine(db_url)
        create_all_tables(engine)

    # Detect endpoint from filename
    endpoint = _detect_endpoint(csv_path)
    respondent = csv_path.stem.split("_")[0]

    if verbose:
        print(f"  Ingesting {csv_path.name}  [{endpoint}]", flush=True)

    run_at = datetime.now(tz=timezone.utc)
    result: dict[str, Any] = {
        "endpoint": endpoint,
        "respondent": respondent,
        "source_file": str(csv_path),
        "rows_raw": 0,
        "rows_ingested": 0,
        "rows_imputed": 0,
        "rows_anomaly": 0,
        "status": "error",
        "error_msg": None,
    }

    try:
        df_raw = _read_raw(csv_path, endpoint)
        result["rows_raw"] = len(df_raw)

        if endpoint == "region-data":
            df_clean, stats = _process_region_data(df_raw, verbose=verbose)
            result["rows_imputed"] = stats["imputed"]
            result["rows_anomaly"] = stats["anomaly"]
            rows_written = _upsert_region_data(df_clean, engine, verbose=verbose)

        elif endpoint == "fuel-type-data":
            df_clean = _process_generic(df_raw)
            rows_written = _upsert_fuel_type(df_clean, engine, verbose=verbose)

        elif endpoint == "interchange-data":
            df_clean = _process_generic(df_raw)
            rows_written = _upsert_interchange(df_clean, engine, verbose=verbose)

        elif endpoint == "region-sub-ba-data":
            df_clean = _process_generic(df_raw)
            rows_written = _upsert_sub_ba(df_clean, engine, verbose=verbose)

        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")

        result["rows_ingested"] = rows_written
        result["status"] = "success"

        if verbose:
            print(
                f"  Done: {rows_written:,} rows ingested"
                + (f"  ({stats['imputed']} imputed, {stats['anomaly']} anomalies)"
                   if endpoint == "region-data" else "")
            )

    except Exception as exc:
        result["error_msg"] = str(exc)
        logger.exception("Ingestion failed for %s", csv_path)
        if verbose:
            print(f"  ERROR: {exc}")

    # Write audit log
    _write_ingest_log(engine, run_at, result)
    return result


def ingest_directory(
    raw_dir: str | Path,
    endpoint: str,
    engine=None,
    db_url: str | None = None,
    pattern: str = "*.csv",
    skip_existing: bool = True,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Ingest all CSVs in a raw data directory for one endpoint.

    Args:
        raw_dir: Directory containing raw CSVs (e.g. data/raw/region-data/)
        endpoint: One of region-data | fuel-type-data | interchange-data | region-sub-ba-data
        engine: SQLAlchemy engine
        skip_existing: Skip BAs already present in DB
        verbose: Print progress

    Returns:
        List of result dicts from ingest_file()
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Directory not found: {raw_dir}")

    if engine is None:
        engine = get_engine(db_url)
        create_all_tables(engine)

    files = sorted(raw_dir.glob(pattern))
    if not files:
        logger.warning("No CSV files found in %s", raw_dir)
        return []

    # Check which BAs are already ingested
    existing_bas: set[str] = set()
    if skip_existing:
        existing_bas = _get_existing_bas(engine, endpoint)

    results = []
    for i, fpath in enumerate(files, 1):
        ba = fpath.stem.split("_")[0]
        if skip_existing and ba in existing_bas:
            if verbose:
                print(f"  [{i:>4}/{len(files)}] {ba:<10} SKIP (already in DB)")
            continue

        if verbose:
            print(f"\n  [{i:>4}/{len(files)}] {ba}", flush=True)

        result = ingest_file(fpath, engine=engine, verbose=verbose)
        results.append(result)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Raw reading
# ─────────────────────────────────────────────────────────────────────────────

def _detect_endpoint(path: Path) -> str:
    name = path.stem.lower()
    for ep in ("region-sub-ba-data", "fuel-type-data", "interchange-data", "region-data"):
        if ep in name:
            return ep
    raise ValueError(f"Cannot detect endpoint from filename: {path.name}")


def _read_raw(csv_path: Path, endpoint: str) -> pd.DataFrame:
    """Read raw CSV with minimal parsing — just timestamps and values."""
    df = pd.read_csv(csv_path, low_memory=False)

    # Parse period → UTC datetime
    df["period"] = pd.to_datetime(
        df["period"], format="%Y-%m-%dT%H", utc=True, errors="coerce"
    )
    invalid = df["period"].isna().sum()
    if invalid > 0:
        logger.warning("Dropped %d rows with unparseable period in %s", invalid, csv_path.name)
        df = df.dropna(subset=["period"])

    # Coerce value to float
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# region-data processing (gap fill + anomaly detection)
# ─────────────────────────────────────────────────────────────────────────────

def _process_region_data(
    df: pd.DataFrame,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean region-data DataFrame.

    For each (respondent, type) series:
        1. Reindex to strict hourly frequency
        2. Interpolate gaps ≤ MAX_GAP_HOURS (mark is_imputed)
        3. Flag anomalies using rolling 168h z-score (mark is_anomaly)

    Returns (cleaned_df, stats_dict)
    """
    records = []
    total_imputed = 0
    total_anomaly = 0

    grouped = df.groupby(["respondent", "type"], sort=False)

    for (respondent, dtype), group in grouped:
        group = group.sort_values("period").copy()

        # Build hourly series
        series = group.set_index("period")["value"]
        series = series[~series.index.duplicated(keep="first")]

        # Reindex to strict hourly
        full_idx = pd.date_range(
            start=series.index.min(),
            end=series.index.max(),
            freq="h",
            tz="UTC",
        )
        series = series.reindex(full_idx)

        # Track which were originally NaN (gaps)
        is_gap = series.isna()

        # Interpolate gaps ≤ MAX_GAP_HOURS
        # Strategy: only interpolate if gap run ≤ MAX_GAP_HOURS
        series_filled = _limited_interpolate(series, max_gap=MAX_GAP_HOURS)
        is_imputed = is_gap & series_filled.notna()
        total_imputed += is_imputed.sum()

        # Anomaly detection on filled series
        rolling_mean = series_filled.rolling(ROLLING_WINDOW, center=True, min_periods=24).mean()
        rolling_std = series_filled.rolling(ROLLING_WINDOW, center=True, min_periods=24).std()
        z_score = (series_filled - rolling_mean).abs() / rolling_std.replace(0, np.nan)
        is_anomaly = (z_score > ANOMALY_STD) & series_filled.notna()
        total_anomaly += is_anomaly.sum()

        # Get metadata from first row
        meta = group.iloc[0]
        respondent_name = meta.get("respondent-name", "")
        type_name = meta.get("type-name", "")

        for period, value in series_filled.items():
            records.append({
                "period": period,
                "respondent": respondent,
                "respondent_name": respondent_name,
                "type": dtype,
                "type_name": type_name,
                "value_mwh": float(value) if pd.notna(value) else None,
                "is_imputed": int(is_imputed.get(period, False)),
                "is_anomaly": int(is_anomaly.get(period, False)),
            })

    result_df = pd.DataFrame(records)
    stats = {"imputed": total_imputed, "anomaly": total_anomaly}
    return result_df, stats


def _limited_interpolate(series: pd.Series, max_gap: int) -> pd.Series:
    """Linear interpolation only for runs of NaN ≤ max_gap length."""
    s = series.copy()
    # Mark gap groups
    is_na = s.isna()
    gap_group = (is_na != is_na.shift()).cumsum()
    gap_sizes = is_na.groupby(gap_group).transform("sum")
    # Only fill gaps within the limit
    fill_mask = is_na & (gap_sizes <= max_gap)
    s_interp = s.interpolate(method="time")
    s[fill_mask] = s_interp[fill_mask]
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Generic processing (other endpoints — no gap filling needed)
# ─────────────────────────────────────────────────────────────────────────────

def _process_generic(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal cleaning for non-modelling endpoints."""
    df = df.drop_duplicates()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Upsert helpers
# ─────────────────────────────────────────────────────────────────────────────

def _upsert_region_data(df: pd.DataFrame, engine, verbose: bool = False) -> int:
    """Upsert region_data rows in chunks. Returns total rows written."""
    total = 0
    for chunk_df in _chunked(df, CHUNK_SIZE):
        rows = chunk_df.to_dict(orient="records")
        with engine.begin() as conn:
            stmt = sqlite_insert(RegionData).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["period", "respondent", "type"],
                set_={
                    "value_mwh": stmt.excluded.value_mwh,
                    "is_imputed": stmt.excluded.is_imputed,
                    "is_anomaly": stmt.excluded.is_anomaly,
                    "respondent_name": stmt.excluded.respondent_name,
                    "type_name": stmt.excluded.type_name,
                },
            )
            conn.execute(stmt)
        total += len(rows)
        if verbose:
            print(f"\r  upserted {total:,} / {len(df):,} rows   ", end="", flush=True)
    if verbose:
        print()
    return total


def _upsert_fuel_type(df: pd.DataFrame, engine, verbose: bool = False) -> int:
    total = 0
    col_map = {"respondent-name": "respondent_name", "type-name": "type_name"}
    df = df.rename(columns=col_map)
    needed = ["period", "respondent", "respondent_name", "fueltype", "type_name", "value"]
    df = df[[c for c in needed if c in df.columns]].rename(columns={"value": "value_mwh"})

    for chunk_df in _chunked(df, CHUNK_SIZE):
        rows = chunk_df.to_dict(orient="records")
        with engine.begin() as conn:
            stmt = sqlite_insert(FuelTypeData).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["period", "respondent", "fueltype"],
                set_={"value_mwh": stmt.excluded.value_mwh},
            )
            conn.execute(stmt)
        total += len(rows)
    return total


def _upsert_interchange(df: pd.DataFrame, engine, verbose: bool = False) -> int:
    total = 0
    col_map = {"fromba-name": "fromba_name", "toba-name": "toba_name"}
    df = df.rename(columns=col_map)
    needed = ["period", "fromba", "fromba_name", "toba", "toba_name", "value"]
    df = df[[c for c in needed if c in df.columns]].rename(columns={"value": "value_mwh"})

    for chunk_df in _chunked(df, CHUNK_SIZE):
        rows = chunk_df.to_dict(orient="records")
        with engine.begin() as conn:
            stmt = sqlite_insert(InterchangeData).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["period", "fromba", "toba"],
                set_={"value_mwh": stmt.excluded.value_mwh},
            )
            conn.execute(stmt)
        total += len(rows)
    return total


def _upsert_sub_ba(df: pd.DataFrame, engine, verbose: bool = False) -> int:
    total = 0
    col_map = {"subba-name": "subba_name"}
    df = df.rename(columns=col_map)
    needed = ["period", "subba", "subba_name", "parent", "value"]
    df = df[[c for c in needed if c in df.columns]].rename(columns={"value": "value_mwh"})

    for chunk_df in _chunked(df, CHUNK_SIZE):
        rows = chunk_df.to_dict(orient="records")
        with engine.begin() as conn:
            stmt = sqlite_insert(SubBaData).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["period", "subba"],
                set_={"value_mwh": stmt.excluded.value_mwh},
            )
            conn.execute(stmt)
        total += len(rows)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Audit log
# ─────────────────────────────────────────────────────────────────────────────

def _write_ingest_log(engine, run_at: datetime, result: dict) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(
                sqlite_insert(IngestLog).values(
                    run_at=run_at,
                    endpoint=result.get("endpoint"),
                    respondent=result.get("respondent"),
                    source_file=result.get("source_file"),
                    rows_raw=result.get("rows_raw", 0),
                    rows_ingested=result.get("rows_ingested", 0),
                    rows_imputed=result.get("rows_imputed", 0),
                    rows_anomaly=result.get("rows_anomaly", 0),
                    status=result.get("status"),
                    error_msg=result.get("error_msg"),
                )
            )
    except Exception as exc:
        logger.warning("Could not write ingest log: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chunked(df: pd.DataFrame, size: int):
    for start in range(0, len(df), size):
        yield df.iloc[start: start + size]


def _get_existing_bas(engine, endpoint: str) -> set[str]:
    """Return set of BA codes already present in the DB for given endpoint."""
    table_map = {
        "region-data": "region_data",
        "fuel-type-data": "fuel_type_data",
        "interchange-data": "interchange_data",
        "region-sub-ba-data": "sub_ba_data",
    }
    col_map = {
        "region-data": "respondent",
        "fuel-type-data": "respondent",
        "interchange-data": "fromba",
        "region-sub-ba-data": "parent",
    }
    table = table_map.get(endpoint)
    col = col_map.get(endpoint)
    if not table or not col:
        return set()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(f"SELECT DISTINCT {col} FROM {table}")).fetchall()
            return {r[0] for r in rows}
    except Exception:
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience query helpers (used by feature engineering + EDA)
# ─────────────────────────────────────────────────────────────────────────────

def load_region_series(
    engine,
    respondent: str,
    dtype: str = "D",
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Load a single (respondent, type) series as a pandas Series indexed by period.

    Args:
        engine: SQLAlchemy engine
        respondent: BA code, e.g. 'MISO'
        dtype: D | DF | NG | TI
        start: ISO date string (optional filter)
        end: ISO date string (optional filter)

    Returns:
        pd.Series with DatetimeIndex (UTC), name = f"{respondent}_{dtype}"
    """
    q = "SELECT period, value_mwh FROM region_data WHERE respondent=:r AND type=:t"
    params: dict = {"r": respondent, "t": dtype}
    if start:
        q += " AND period >= :start"
        params["start"] = start
    if end:
        q += " AND period <= :end"
        params["end"] = end
    q += " ORDER BY period"

    with engine.connect() as conn:
        df = pd.read_sql_query(text(q), conn, params=params, parse_dates=["period"])

    df["period"] = pd.to_datetime(df["period"], utc=True)
    s = df.set_index("period")["value_mwh"]
    s.name = f"{respondent}_{dtype}"
    return s


def load_wide(
    engine,
    respondent: str,
    types: list[str] = ("D", "DF", "NG", "TI"),
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load multiple types for one BA as a wide DataFrame.

    Returns DataFrame with columns D, DF, NG, TI indexed by period (UTC).
    """
    frames = {
        t: load_region_series(engine, respondent, dtype=t, start=start, end=end)
        for t in types
    }
    return pd.DataFrame(frames)
