#!/usr/bin/env python3
"""fetch_weather.py — Download weather data from Open-Meteo for all BAs.

Usage:
    python scripts/fetch_weather.py
    python scripts/fetch_weather.py --ba MISO PJM ERCO
    python scripts/fetch_weather.py --start 2022-01-01 --end 2023-12-31
    python scripts/fetch_weather.py --force
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config_loader import get_ba_codes, load_config
from src.data.database import get_engine, create_all_tables
from src.data.ba_coordinates import get_ba_coordinates, validate_coordinates
from src.data.weather_client import fetch_weather_for_all_bas, save_weather_to_db


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch weather data from Open-Meteo for Balancing Authorities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fetch_weather.py
  python scripts/fetch_weather.py --ba MISO PJM
  python scripts/fetch_weather.py --start 2022-01-01 --end 2023-12-31
  python scripts/fetch_weather.py --force
        """,
    )
    parser.add_argument(
        "--ba",
        nargs="+",
        default=["ALL"],
        metavar="CODE",
        help="BA codes to process, or ALL (default: ALL)",
    )
    parser.add_argument(
        "--start",
        default="2022-01-01",
        metavar="DATE",
        help="Start date (YYYY-MM-DD) (default: 2022-01-01)",
    )
    parser.add_argument(
        "--end",
        default="2023-12-31",
        metavar="DATE",
        help="End date (YYYY-MM-DD) (default: 2023-12-31)",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/raw/weather",
        metavar="PATH",
        help="Cache directory for weather parquet files (default: data/raw/weather)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh (ignore cache)",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        metavar="URL",
        help="Database URL (default: from env or data/energy_forecasting.db)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    
    # Setup paths
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config and get BA list
    cfg = load_config()
    if args.ba == ["ALL"]:
        ba_codes = get_ba_codes(cfg)
    else:
        ba_codes = args.ba
    
    # Validate all BAs have coordinates
    is_valid, missing = validate_coordinates(ba_codes)
    if not is_valid:
        print(f"Error: Missing coordinates for BAs: {missing}")
        return 1
    
    # Build coordinates dict
    coordinates = {}
    for code in ba_codes:
        coord = get_ba_coordinates(code)
        coordinates[code] = {"lat": coord["lat"], "lon": coord["lon"]}
    
    print(f"\nFetching weather for {len(ba_codes)} BA(s) from {args.start} to {args.end}")
    print("=" * 70)
    
    # Fetch weather data
    df = fetch_weather_for_all_bas(
        ba_codes=ba_codes,
        coordinates=coordinates,
        start_date=args.start,
        end_date=args.end,
        cache_dir=cache_dir,
        force_refresh=args.force,
    )
    
    if df.empty:
        print("No weather data fetched!")
        return 1
    
    print(f"\nFetched {len(df):,} total rows")
    
    # Save to database
    engine = get_engine(args.db_url)
    create_all_tables(engine)
    
    rows_inserted = save_weather_to_db(df, engine)
    print(f"Saved {rows_inserted:,} rows to database")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
