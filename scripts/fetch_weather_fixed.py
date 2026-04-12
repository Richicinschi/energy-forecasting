#!/usr/bin/env python3
"""
Robust weather data fetcher with verification.

Fetches weather data from Open-Meteo for all BAs, verifies completeness,
and only proceeds to next BA when data is confirmed valid.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ba_coordinates import BA_COORDINATES, get_all_ba_coordinates
from src.data.config_loader import load_config
from src.data.database import get_engine, create_all_tables
from src.data.weather_client import save_weather_to_db

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fetch_weather.log')
    ]
)
logger = logging.getLogger(__name__)


def verify_weather_data(
    df: Optional[pd.DataFrame],
    ba_code: str,
    expected_start: pd.Timestamp,
    expected_end: pd.Timestamp
) -> Tuple[bool, str]:
    """
    Verify downloaded weather data is complete and valid.
    
    Returns:
        (is_valid, message)
    """
    if df is None:
        return False, "DataFrame is None"
    
    if len(df) == 0:
        return False, "Empty dataframe"
    
    # Check date range
    actual_start = df.index.min()
    actual_end = df.index.max()
    
    # Allow 7-day buffer for missing data at edges
    start_buffer = expected_start + pd.Timedelta(days=7)
    end_buffer = expected_end - pd.Timedelta(days=7)
    
    if actual_start > start_buffer:
        return False, f"Start date too late: {actual_start} > {start_buffer}"
    
    if actual_end < end_buffer:
        return False, f"End date too early: {actual_end} < {end_buffer}"
    
    # Check core columns exist
    core_cols = ['temp_2m', 'dewpoint_2m', 'windspeed_10m']
    for col in core_cols:
        if col not in df.columns:
            return False, f"Missing column: {col}"
    
    # Check for excessive NaN (>50%)
    for col in core_cols:
        nan_pct = df[col].isna().sum() / len(df)
        if nan_pct > 0.5:
            return False, f"Too many NaN in {col}: {nan_pct:.1%}"
    
    # Check expected row count (hourly data)
    expected_hours = (expected_end - expected_start).total_seconds() / 3600
    actual_hours = len(df)
    
    if actual_hours < expected_hours * 0.9:  # Allow 10% missing
        return False, f"Too few rows: {actual_hours} < {expected_hours * 0.9:.0f} expected"
    
    logger.info(f"  [OK] Date range: {actual_start} to {actual_end}")
    logger.info(f"  [OK] Rows: {len(df):,}")
    logger.info(f"  [OK] Core columns valid")
    
    return True, "Valid"


def fetch_weather_with_retry(
    ba_code: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    max_retries: int = 3,
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """
    Fetch weather data with retry logic and verification.
    
    Returns verified DataFrame or None if all retries fail.
    """
    import requests
    
    expected_start = pd.Timestamp(start_date, tz='UTC')
    expected_end = pd.Timestamp(end_date, tz='UTC')
    
    cache_file = cache_dir / f"{ba_code}.parquet"
    
    for attempt in range(max_retries):
        try:
            logger.info(f"  Fetch attempt {attempt + 1}/{max_retries}...")
            
            # Build Open-Meteo URL
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "hourly": [
                    "temperature_2m",
                    "dewpoint_2m",
                    "windspeed_10m",
                    "shortwave_radiation",
                    "cloudcover",
                    "precipitation"
                ],
                "timezone": "UTC"
            }
            
            response = requests.get(url, params=params, timeout=60)
            
            # Always sleep after request to respect rate limits
            time.sleep(4.0)
            
            if response.status_code == 429:
                wait_time = 60 + attempt * 30  # Longer backoff
                logger.warning(f"  Rate limited (429), waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            hourly = data.get('hourly', {})
            if not hourly:
                logger.error(f"  No hourly data in response")
                continue
            
            # Build DataFrame
            df = pd.DataFrame({
                'temp_2m': hourly.get('temperature_2m'),
                'dewpoint_2m': hourly.get('dewpoint_2m'),
                'windspeed_10m': hourly.get('windspeed_10m'),
                'solar_irradiance': hourly.get('shortwave_radiation'),
                'cloudcover': hourly.get('cloudcover'),
                'precipitation': hourly.get('precipitation')
            })
            
            # Parse timestamps
            df.index = pd.to_datetime(hourly.get('time'), utc=True)
            
            # Verify data
            is_valid, msg = verify_weather_data(df, ba_code, expected_start, expected_end)
            
            if is_valid:
                # Save to cache
                df.to_parquet(cache_file)
                logger.info(f"  [OK] Saved to {cache_file}")
                return df
            else:
                logger.warning(f"  Verification failed: {msg}")
                
        except Exception as e:
            logger.error(f"  Fetch error: {e}")
        
        # Exponential backoff
        wait_time = 4 ** (attempt + 1)
        logger.info(f"  Retrying in {wait_time}s...")
        time.sleep(wait_time)
    
    logger.error(f"  Failed after {max_retries} attempts")
    return None


def fetch_weather_for_ba(
    ba_code: str,
    coordinates: dict,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    force: bool = False,
    verbose: bool = False
) -> bool:
    """
    Fetch weather for a single BA with verification.
    
    Returns True if successful (cached or fetched), False otherwise.
    """
    cache_file = cache_dir / f"{ba_code}.parquet"
    
    lat = coordinates['lat']
    lon = coordinates['lon']
    name = coordinates.get('name', ba_code)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {ba_code} - {name}")
    logger.info(f"Coordinates: {lat:.4f}, {lon:.4f}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Step 1: Check cache if not forcing
    if not force and cache_file.exists():
        logger.info(f"  Cache exists: {cache_file}")
        
        try:
            df = pd.read_parquet(cache_file)
            expected_start = pd.Timestamp(start_date, tz='UTC')
            expected_end = pd.Timestamp(end_date, tz='UTC')
            
            is_valid, msg = verify_weather_data(df, ba_code, expected_start, expected_end)
            
            if is_valid:
                logger.info(f"  [OK] Cache valid, skipping fetch")
                return True
            else:
                logger.warning(f"  Cache invalid: {msg}")
                logger.info(f"  Will refetch...")
        except Exception as e:
            logger.warning(f"  Cache read error: {e}")
            logger.info(f"  Will refetch...")
    
    # Step 2: Fetch from API
    logger.info(f"  Fetching from Open-Meteo...")
    
    df = fetch_weather_with_retry(
        ba_code, lat, lon, start_date, end_date,
        cache_dir, max_retries=3, verbose=verbose
    )
    
    if df is not None:
        logger.info(f"  [OK] Successfully fetched {len(df):,} rows")
        return True
    else:
        logger.error(f"  [FAIL] Failed to fetch {ba_code}")
        return False


def ingest_cache_to_db(ba_codes: list, cache_dir: Path, engine) -> dict:
    """Load cached weather files into database."""
    from sqlalchemy import text
    
    results = {"success": 0, "failed": 0, "rows": 0}
    
    for ba_code in ba_codes:
        cache_file = cache_dir / f"{ba_code}.parquet"
        
        if not cache_file.exists():
            logger.warning(f"{ba_code}: Cache file not found")
            results["failed"] += 1
            continue
        
        try:
            # Load cache
            df = pd.read_parquet(cache_file)
            
            if df.empty:
                logger.warning(f"{ba_code}: Cache file empty")
                results["failed"] += 1
                continue
            
            # Add respondent column if not present
            if 'respondent' not in df.columns:
                df['respondent'] = ba_code
            
            # Reset index to make period a column
            if df.index.name == 'period' or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            
            # Ensure period column exists
            if 'period' not in df.columns and 'time' in df.columns:
                df = df.rename(columns={'time': 'period'})
            
            # Delete existing data for this BA
            with engine.connect() as conn:
                conn.execute(text("DELETE FROM weather_data WHERE respondent = :r"), {"r": ba_code})
                conn.commit()
            
            # Save to database
            rows = save_weather_to_db(df, engine)
            results["rows"] += rows
            results["success"] += 1
            logger.info(f"{ba_code}: Ingested {rows:,} rows")
            
        except Exception as e:
            logger.error(f"{ba_code}: Ingest failed - {e}")
            results["failed"] += 1
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Fetch weather data with verification')
    parser.add_argument('--ba', nargs='+', default=['ALL'],
                       help='BA codes to fetch (or ALL)')
    parser.add_argument('--start', type=str, default='2019-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2026-04-09',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--force', action='store_true',
                       help='Force refetch even if cache exists')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--delay', type=float, default=4.0,
                       help='Seconds between API calls (default: 4.0)')
    parser.add_argument('--ingest', action='store_true',
                       help='Ingest cache files to database (no fetch)')
    
    args = parser.parse_args()
    
    # Setup cache directory
    cache_dir = Path('data/raw/weather')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Get BA list
    if 'ALL' in args.ba:
        config = load_config()
        # Extract enabled BA codes from config (list of dicts)
        ba_codes = [ba['code'] for ba in config['balancing_authorities'] if ba.get('enabled', True)]
        logger.info(f"Fetching weather for {len(ba_codes)} enabled BAs from config")
    else:
        ba_codes = args.ba
        logger.info(f"Fetching weather for {len(ba_codes)} specified BAs")
    
    # Get coordinates
    all_coords = get_all_ba_coordinates()
    
    # Track results
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # Fetch for each BA
    for i, ba_code in enumerate(ba_codes):
        if ba_code not in all_coords:
            logger.error(f"No coordinates for {ba_code}, skipping")
            fail_count += 1
            continue
        
        success = fetch_weather_for_ba(
            ba_code,
            all_coords[ba_code],
            args.start,
            args.end,
            cache_dir,
            force=args.force,
            verbose=args.verbose
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        # Progress
        logger.info(f"\n  Progress: {i+1}/{len(ba_codes)} | "
                   f"Success: {success_count} | Failed: {fail_count}")
        
        # Rate limiting (except after last BA)
        if i < len(ba_codes) - 1:
            delay = 6.0 + (hash(ba_code) % 100) / 100  # Minimum 6s between BAs
            logger.info(f"  Waiting {delay:.1f}s before next BA...")
            time.sleep(delay)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("FETCH COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total BAs: {len(ba_codes)}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Cache hits: {skip_count}")
    
    if fail_count > 0:
        logger.warning("Some BAs failed. Check logs and retry with --force if needed.")
        sys.exit(1)
    else:
        logger.info("All BAs successfully fetched!")
        sys.exit(0)


if __name__ == '__main__':
    main()
