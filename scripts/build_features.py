#!/usr/bin/env python3
"""
build_features.py — Build feature matrices for energy load forecasting.

Reads from SQLite (populated by the ingest pipeline) and writes per-BA
Parquet feature files to data/processed/features/.

Usage:
    # All enabled BAs (default)
    python scripts/build_features.py

    # Specific BAs only
    python scripts/build_features.py --ba MISO PJM ERCO

    # Skip the combined ALL_features.parquet
    python scripts/build_features.py --no-combined

    # Custom output directory
    python scripts/build_features.py --output-dir /data/features

    # Custom database URL
    python scripts/build_features.py --db-url sqlite:///data/custom.db
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config_loader import get_ba_codes, load_config
from src.data.database import get_engine
from src.features.engineer import build_features_all


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build feature matrices from EIA-930 SQLite data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_features.py
  python scripts/build_features.py --ba MISO PJM
  python scripts/build_features.py --no-combined
        """,
    )
    parser.add_argument(
        "--ba",
        nargs="+",
        default=["ALL"],
        metavar="CODE",
        help="BA codes to process, or ALL (default: ALL enabled BAs from config)",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        metavar="URL",
        help="SQLAlchemy DB URL (default: data/energy.db relative to project root)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="Output directory for Parquet files (default: data/processed/features/)",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Skip creating ALL_features.parquet",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).resolve().parents[1] / "data" / "processed" / "features"
    )

    engine = get_engine(args.db_url)
    cfg = load_config()

    if args.ba == ["ALL"]:
        ba_codes = get_ba_codes(cfg)
    else:
        ba_codes = args.ba

    print(f"\nBuilding features for {len(ba_codes)} BA(s)  ->  {output_dir}")
    print("=" * 70)

    build_features_all(
        engine=engine,
        ba_codes=ba_codes,
        output_dir=output_dir,
        combined=not args.no_combined,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
