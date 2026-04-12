#!/usr/bin/env python3
"""Fetch only the missing weather fields one at a time and merge with existing cache."""

import sys
from pathlib import Path
import pandas as pd
import requests
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.ba_coordinates import get_all_ba_coordinates
from src.data.config_loader import load_config


def fetch_single_field(ba_code, lat, lon, start_date, end_date, field_name, max_retries=5):
    """Fetch a single weather field."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": field_name,
        "timezone": "UTC",
    }
    
    for attempt in range(max_retries):
        # Longer delay for single field requests
        time.sleep(6.0)
        
        try:
            resp = requests.get(url, params=params, timeout=60)
            
            if resp.status_code == 429:
                wait_time = 90 + attempt * 30  # Start at 90s
                print(f"      Rate limited (429), waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            resp.raise_for_status()
            data = resp.json()
            hourly = data.get("hourly", {})
            
            return hourly.get(field_name)
        except Exception as e:
            print(f"      Attempt {attempt+1} failed - {e}")
            time.sleep(15)
    
    return None


def main():
    cache_dir = Path("data/raw/weather")
    config = load_config()
    ba_codes = [ba['code'] for ba in config['balancing_authorities'] if ba.get('enabled', True)]
    coords = get_all_ba_coordinates()
    
    # Fields to fetch one by one
    fields_to_fetch = [
        ("apparent_temperature", "apparent_temp"),
        ("relative_humidity_2m", "relative_humidity"),
        ("surface_pressure", "surface_pressure"),
    ]
    
    print(f"Updating {len(ba_codes)} BAs with 3 additional fields (one at a time)...")
    print("=" * 60)
    
    for i, ba_code in enumerate(ba_codes, 1):
        cache_file = cache_dir / f"{ba_code}.parquet"
        
        if not cache_file.exists():
            print(f"[{i}/{len(ba_codes)}] {ba_code}: No cache file, skipping")
            continue
        
        print(f"\n[{i}/{len(ba_codes)}] {ba_code}: Loading existing cache...")
        df = pd.read_parquet(cache_file)
        
        # Check which fields are missing
        missing_fields = []
        for api_name, col_name in fields_to_fetch:
            if col_name not in df.columns:
                missing_fields.append((api_name, col_name))
        
        if not missing_fields:
            print(f"  Already has all new columns, skipping")
            continue
        
        coord = coords.get(ba_code, {})
        if not coord:
            print(f"  No coordinates, skipping")
            continue
        
        # Fetch each missing field one by one
        for api_name, col_name in missing_fields:
            print(f"  Fetching {col_name}...")
            
            values = fetch_single_field(
                ba_code, coord["lat"], coord["lon"],
                "2019-01-01", "2026-04-12",
                api_name
            )
            
            if values is None:
                print(f"    Failed to fetch {col_name}, will retry next run")
                continue
            
            df[col_name] = values
            print(f"    [OK] Added {col_name}")
        
        # Save back (even if some fields failed)
        df.to_parquet(cache_file)
        print(f"  [OK] Saved {len(df)} rows")
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
