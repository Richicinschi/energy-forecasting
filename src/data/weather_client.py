"""
weather_client.py — Open-Meteo API client for historical weather data.

Fetches hourly weather data for Balancing Authority regions including:
  - Temperature (2m)
  - Dewpoint (2m)
  - Wind speed (10m)
  - Shortwave solar radiation
  - Cloud cover
  - Precipitation

Open-Meteo API docs: https://open-meteo.com/en/docs/historical-weather-api

Usage:
    from src.data.weather_client import fetch_weather_for_ba, fetch_weather_for_all_bas

    # Fetch for single BA
    df = fetch_weather_for_ba(
        ba_code="MISO",
        lat=39.8,
        lon=-87.0,
        start_date="2022-01-01",
        end_date="2023-12-31",
    )

    # Fetch for multiple BAs
    coordinates = {
        "MISO": {"lat": 39.8, "lon": -87.0},
        "PJM": {"lat": 40.0, "lon": -77.0},
    }
    df_all = fetch_weather_for_all_bas(
        ba_codes=["MISO", "PJM"],
        coordinates=coordinates,
        start_date="2022-01-01",
        end_date="2023-12-31",
    )
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo API column mapping: API name -> DataFrame column name
COLUMN_MAPPING = {
    "temperature_2m": "temp_2m",
    "apparent_temperature": "apparent_temp",
    "dewpoint_2m": "dewpoint_2m",
    "relative_humidity_2m": "relative_humidity",
    "windspeed_10m": "windspeed_10m",
    "shortwave_radiation": "solar_irradiance",
    "cloudcover": "cloudcover",
    "precipitation": "precipitation",
    "surface_pressure": "surface_pressure",
}


def _create_session(retry_attempts: int = 3, retry_backoff: float = 0.5) -> requests.Session:
    """Create a requests Session with retry logic.

    Args:
        retry_attempts: Number of times to retry a failed request.
        retry_backoff: Exponential backoff multiplier between retries.

    Returns:
        Configured requests Session.
    """
    session = requests.Session()
    retry = Retry(
        total=retry_attempts,
        backoff_factor=retry_backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_weather_for_ba(
    ba_code: str,
    lat: float,
    lon: float,
    start_date: str,  # YYYY-MM-DD
    end_date: str,    # YYYY-MM-DD
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch weather data for a single Balancing Authority from Open-Meteo.

    Args:
        ba_code: Balancing Authority code, e.g. "MISO", "PJM", "ERCO".
        lat: Latitude of the BA centroid.
        lon: Longitude of the BA centroid.
        start_date: Start date in "YYYY-MM-DD" format (inclusive).
        end_date: End date in "YYYY-MM-DD" format (inclusive).
        cache_dir: Optional directory to cache parquet files.
        force_refresh: If True, ignore cache and fetch fresh data.

    Returns:
        DataFrame with columns:
            temp_2m, dewpoint_2m, windspeed_10m, solar_irradiance,
            cloudcover, precipitation
        With a UTC DatetimeIndex.
        Returns empty DataFrame if the request fails.
    """
    # Check cache first
    if cache_dir is not None and not force_refresh:
        cache_path = Path(cache_dir) / f"{ba_code}.parquet"
        if cache_path.exists():
            logger.info("Loading cached weather data for %s from %s", ba_code, cache_path)
            try:
                df = pd.read_parquet(cache_path)
                logger.info("Loaded %d cached rows for %s", len(df), ba_code)
                return df
            except Exception as exc:
                logger.warning("Failed to load cache for %s: %s", ba_code, exc)

    # Build API URL with parameters
    hourly_vars = ",".join([
        "temperature_2m",
        "apparent_temperature",
        "dewpoint_2m",
        "relative_humidity_2m",
        "windspeed_10m",
        "shortwave_radiation",
        "cloudcover",
        "precipitation",
        "surface_pressure",
    ])

    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_vars,
        "timezone": "UTC",
    }

    logger.info(
        "Fetching weather for %s (%.4f, %.4f) from %s to %s",
        ba_code, lat, lon, start_date, end_date
    )

    try:
        session = _create_session()
        resp = session.get(OPEN_METEO_URL, params=params, timeout=30)
        
        # Rate limiting - Open-Meteo allows ~600 requests/minute
        # We use 1.0s to be safe and avoid 429 errors
        time.sleep(3.5)

        if resp.status_code != 200:
            logger.warning(
                "Open-Meteo API returned HTTP %d for %s: %s",
                resp.status_code, ba_code, resp.text[:500]
            )
            return pd.DataFrame()

        data = resp.json()

        if "hourly" not in data:
            logger.warning("No hourly data in response for %s", ba_code)
            return pd.DataFrame()

        hourly = data["hourly"]
        
        if not hourly.get("time"):
            logger.warning("Empty time array in response for %s", ba_code)
            return pd.DataFrame()

        # Build DataFrame from hourly data
        df_data = {"time": hourly["time"]}
        for api_col, df_col in COLUMN_MAPPING.items():
            if api_col in hourly:
                df_data[df_col] = hourly[api_col]
            else:
                logger.warning("Missing column %s in response for %s", api_col, ba_code)
                df_data[df_col] = [None] * len(hourly["time"])

        df = pd.DataFrame(df_data)

        # Convert time to datetime and set as index
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
        df.index.name = "period"

        logger.info("Fetched %d weather rows for %s", len(df), ba_code)

        # Save to cache if cache_dir provided
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"{ba_code}.parquet"
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path)
                logger.info("Cached weather data for %s to %s", ba_code, cache_path)
            except Exception as exc:
                logger.warning("Failed to cache weather data for %s: %s", ba_code, exc)

        return df

    except requests.RequestException as exc:
        logger.warning("Request failed for %s: %s", ba_code, exc)
        return pd.DataFrame()
    except Exception as exc:
        logger.warning("Unexpected error fetching weather for %s: %s", ba_code, exc)
        return pd.DataFrame()


def fetch_weather_for_all_bas(
    ba_codes: list[str],
    coordinates: dict[str, dict],  # {ba_code: {"lat": float, "lon": float}}
    start_date: str,
    end_date: str,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch weather data for multiple Balancing Authorities.

    Args:
        ba_codes: List of BA codes to fetch, e.g. ["MISO", "PJM", "ERCO"].
        coordinates: Dictionary mapping BA codes to coordinate dicts with
            "lat" and "lon" keys.
        start_date: Start date in "YYYY-MM-DD" format (inclusive).
        end_date: End date in "YYYY-MM-DD" format (inclusive).
        cache_dir: Optional directory to cache parquet files.
        force_refresh: If True, ignore cache and fetch fresh data.

    Returns:
        Combined DataFrame with all BAs, including a 'respondent' column
        with the BA code. Returns empty DataFrame if no data was fetched.

    Raises:
        ValueError: If a BA code is missing from coordinates or if coordinates
            dict is missing 'lat' or 'lon' keys.
    """
    frames: list[pd.DataFrame] = []
    total = len(ba_codes)

    for i, ba_code in enumerate(ba_codes, 1):
        logger.info("Fetching %s (%d/%d)...", ba_code, i, total)

        # Validate coordinates
        if ba_code not in coordinates:
            raise ValueError(
                f"BA code '{ba_code}' not found in coordinates dictionary"
            )
        
        coord = coordinates[ba_code]
        if "lat" not in coord or "lon" not in coord:
            raise ValueError(
                f"Coordinates for '{ba_code}' must contain 'lat' and 'lon' keys"
            )

        try:
            df = fetch_weather_for_ba(
                ba_code=ba_code,
                lat=coord["lat"],
                lon=coord["lon"],
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                force_refresh=force_refresh,
            )
            
            if not df.empty:
                # Add respondent column with BA code
                df = df.reset_index()
                df["respondent"] = ba_code
                frames.append(df)
                logger.info("[%s] OK — %d rows", ba_code, len(df))
            else:
                logger.warning("[%s] WARNING — no data returned", ba_code)
        
        except Exception as exc:
            logger.error("Failed to fetch weather for %s: %s", ba_code, exc)

    if not frames:
        return pd.DataFrame()
    
    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Combined weather data: %d rows for %d BAs",
        len(combined), len(frames)
    )
    return combined


def save_weather_to_db(df: pd.DataFrame, engine) -> int:
    """Save weather DataFrame to SQLite database.

    Args:
        df: DataFrame with weather data. Expected columns:
            period, respondent, temp_2m, dewpoint_2m, windspeed_10m,
            solar_irradiance, cloudcover, precipitation
        engine: SQLAlchemy engine or connection.

    Returns:
        Number of rows inserted into the database.
    """
    if df.empty:
        logger.warning("Empty DataFrame, nothing to save to database")
        return 0

    # Ensure period column exists (might be index)
    df_to_save = df.copy()
    if df_to_save.index.name == "period":
        df_to_save = df_to_save.reset_index()
    elif "period" not in df_to_save.columns:
        raise ValueError("DataFrame must have 'period' column or DatetimeIndex named 'period'")

    # Define expected columns for the database (6 core columns)
    expected_columns = [
        "period",
        "respondent",
        "temp_2m",
        "dewpoint_2m",
        "windspeed_10m",
        "solar_irradiance",
        "cloudcover",
        "precipitation",
    ]

    # Filter to only expected columns, warn about missing ones
    available_cols = [col for col in expected_columns if col in df_to_save.columns]
    missing_cols = [col for col in expected_columns if col not in df_to_save.columns]
    
    if missing_cols:
        logger.warning("Missing columns in DataFrame: %s", missing_cols)
    
    df_to_save = df_to_save[available_cols]

    try:
        rows_inserted = df_to_save.to_sql(
            name="weather_data",
            con=engine,
            if_exists="append",
            method="multi",
            index=False,
        )
        logger.info("Saved %d weather rows to database", rows_inserted)
        return int(rows_inserted) if rows_inserted is not None else 0
    except Exception as exc:
        logger.error("Failed to save weather data to database: %s", exc)
        raise
