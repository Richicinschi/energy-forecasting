#!/usr/bin/env python3
"""
ingest.py — CLI for ingesting EIA-930 raw CSVs into the SQLite database.

Usage:
    # Ingest a specific file
    python scripts/ingest.py --file data/raw/region-data/MISO_20190101_20260409_region-data.csv

    # Ingest all files for one endpoint
    python scripts/ingest.py --endpoint region-data

    # Ingest all endpoints
    python scripts/ingest.py --endpoint all

    # Force re-ingest even if BA already in DB
    python scripts/ingest.py --endpoint region-data --force

    # Show DB table counts
    python scripts/ingest.py --status
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.data.database import create_all_tables, get_engine, get_table_counts
from src.data.ingest import ingest_directory, ingest_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_cli")

ENDPOINT_DIRS = {
    "region-data":       "data/raw/region-data",
    "fuel-type-data":    "data/raw/fuel-type-data",
    "interchange-data":  "data/raw/interchange-data",
    "region-sub-ba-data": "data/raw/region-sub-ba-data",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Ingest EIA-930 raw CSVs into SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", "-f", help="Path to a single raw CSV to ingest")
    group.add_argument(
        "--endpoint", "-e", nargs="+",
        choices=list(ENDPOINT_DIRS.keys()) + ["all"],
        help="Endpoint(s) to ingest, or 'all'",
    )
    parser.add_argument("--db-url", default=None, help="Database URL override")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if already in DB")
    parser.add_argument("--status", action="store_true", help="Show DB table counts and exit")
    return parser.parse_args(argv)


def print_status(engine) -> None:
    counts = get_table_counts(engine)
    print()
    print("=" * 50)
    print("  Database Table Counts")
    print("=" * 50)
    for table, count in counts.items():
        print(f"  {table:<25} {count:>12,}")
    print("=" * 50)
    print()


def main(argv=None) -> int:
    args = parse_args(argv)
    engine = get_engine(args.db_url)
    create_all_tables(engine)

    if args.status:
        print_status(engine)
        return 0

    t0 = time.time()

    if args.file:
        result = ingest_file(args.file, engine=engine, verbose=True)
        ok = result["status"] == "success"
        print(f"\n{'OK' if ok else 'FAILED'}: {result['rows_ingested']:,} rows ingested")
        return 0 if ok else 1

    endpoints = args.endpoint or ["region-data"]
    if "all" in endpoints:
        endpoints = list(ENDPOINT_DIRS.keys())

    grand_total = 0
    grand_failed = []

    for endpoint in endpoints:
        raw_dir = Path(ENDPOINT_DIRS[endpoint])
        if not raw_dir.exists():
            print(f"\nWARNING: Directory not found — {raw_dir}  (skipping {endpoint})")
            continue

        files = list(raw_dir.glob("*.csv"))
        print(f"\n{'='*60}")
        print(f"  ENDPOINT: {endpoint}  ({len(files)} files)")
        print(f"{'='*60}")

        results = ingest_directory(
            raw_dir=raw_dir,
            endpoint=endpoint,
            engine=engine,
            skip_existing=not args.force,
            verbose=True,
        )

        done = sum(1 for r in results if r["status"] == "success")
        failed = [r["respondent"] for r in results if r["status"] == "error"]
        rows = sum(r["rows_ingested"] for r in results if r["status"] == "success")
        grand_total += rows
        grand_failed.extend(failed)

        print(f"\n  {endpoint}: {done} ingested, {len(failed)} failed, {rows:,} rows")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"  Ingest complete in {elapsed:.0f}s")
    print(f"  Total rows written: {grand_total:,}")
    if grand_failed:
        print(f"  Failed BAs: {', '.join(grand_failed)}")
    print()
    print_status(engine)

    return 0 if not grand_failed else 1


if __name__ == "__main__":
    sys.exit(main())
