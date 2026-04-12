#!/usr/bin/env python3
"""Clear and re-fetch weather data for full date range."""
import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.database import get_engine
from sqlalchemy import text
from src.data.config_loader import load_config, get_ba_codes
from src.data.ba_coordinates import get_ba_coordinates
from src.data.weather_client import fetch_weather_for_all_bas, save_weather_to_db

def main():
    print("="*70)
    print("Re-fetching weather data for full date range (2019-2026)")
    print("="*70)
    
    # Step 1: Clear existing weather data from database
    print("\n1. Clearing existing weather data from SQLite...")
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM weather_data"))
        conn.commit()
        result = conn.execute(text("SELECT COUNT(*) FROM weather_data"))
        print(f"   Weather table now has {result.scalar()} rows")
    
    # Step 2: Delete cached parquet files
    print("\n2. Deleting cached weather parquet files...")
    cache_dir = Path("data/raw/weather")
    if cache_dir.exists():
        files_deleted = 0
        for f in cache_dir.glob("*.parquet"):
            f.unlink()
            files_deleted += 1
        print(f"   Deleted {files_deleted} cached files")
    
    # Step 3: Fetch weather for full date range
    print("\n3. Fetching weather from Open-Meteo (2019-01-01 to 2026-04-09)...")
    cfg = load_config()
    ba_codes = get_ba_codes(cfg)
    
    # Build coordinates dict
    coordinates = {}
    for code in ba_codes:
        coord = get_ba_coordinates(code)
        coordinates[code] = {"lat": coord["lat"], "lon": coord["lon"]}
    
    print(f"   Fetching for {len(ba_codes)} BAs...")
    print(f"   This will take approximately {len(ba_codes) * 1.5:.0f} minutes...")
    
    df = fetch_weather_for_all_bas(
        ba_codes=ba_codes,
        coordinates=coordinates,
        start_date="2019-01-01",
        end_date="2026-04-09",
        cache_dir=cache_dir,
        force_refresh=True,
    )
    
    if df.empty:
        print("ERROR: No weather data fetched!")
        return 1
    
    print(f"\n   Fetched {len(df):,} total rows")
    
    # Step 4: Save to database
    print("\n4. Saving to SQLite database...")
    rows_inserted = 0
    chunk_size = 100
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        chunk.to_sql(
            'weather_data',
            engine,
            if_exists='append',
            index=False,
            method='multi'
        )
        rows_inserted += len(chunk)
        if i % 10000 == 0:
            print(f"   Saved {rows_inserted:,} rows...")
    
    print(f"\n   Total rows saved: {rows_inserted:,}")
    
    # Verify
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM weather_data"))
        count = result.scalar()
        result = conn.execute(text("SELECT MIN(period), MAX(period) FROM weather_data"))
        min_p, max_p = result.fetchone()
        print(f"\n5. Verification:")
        print(f"   Total rows in database: {count:,}")
        print(f"   Date range: {min_p} to {max_p}")
    
    print("\n" + "="*70)
    print("Weather data refresh complete!")
    print("Next: Run scripts/build_features.py to rebuild feature matrices")
    print("="*70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
