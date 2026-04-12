#!/usr/bin/env python3
"""Resume fetching weather data with better rate limiting."""
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.config_loader import load_config, get_ba_codes
from src.data.ba_coordinates import get_ba_coordinates
from src.data.weather_client import fetch_weather_for_ba

def main():
    print("="*70)
    print("Resuming weather fetch with 2s delay (better rate limiting)")
    print("="*70)
    
    cache_dir = Path("data/raw/weather")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of BAs that still need fetching
    cached = {p.stem for p in cache_dir.glob("*.parquet")}
    print(f"Already cached: {len(cached)} BAs")
    
    cfg = load_config()
    all_bas = get_ba_codes(cfg)
    
    to_fetch = [ba for ba in all_bas if ba not in cached]
    print(f"Still need to fetch: {len(to_fetch)} BAs")
    print(f"BAs to fetch: {to_fetch}")
    print()
    
    # Fetch missing BAs one at a time with 2s delay
    failed = []
    for i, ba in enumerate(to_fetch, 1):
        coord = get_ba_coordinates(ba)
        print(f"[{i}/{len(to_fetch)}] Fetching {ba} ({coord['name']})...")
        
        try:
            df = fetch_weather_for_ba(
                ba_code=ba,
                lat=coord["lat"],
                lon=coord["lon"],
                start_date="2019-01-01",
                end_date="2026-04-09",
                cache_dir=cache_dir,
                force_refresh=False,  # Don't force, use cache if exists
            )
            
            if df.empty:
                print(f"  WARNING: No data returned for {ba}")
                failed.append(ba)
            else:
                print(f"  SUCCESS: {len(df):,} rows")
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(ba)
        
        # 2 second delay between requests
        if i < len(to_fetch):
            time.sleep(2.0)
    
    print()
    print("="*70)
    print(f"Fetch complete! Cached: {len(cached) + len(to_fetch) - len(failed)}/{len(all_bas)}")
    if failed:
        print(f"Failed: {failed}")
    print("="*70)
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
