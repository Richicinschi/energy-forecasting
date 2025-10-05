#!/usr/bin/env python3
"""
fetch_data.py — CLI for fetching EIA-930 electricity demand data.

Fetches actual demand (D) and EIA's own forecast (DF) for one or all
Balancing Authorities and saves raw CSVs to data/raw/.

Usage examples:
    # Single region, 2 years of data
    python scripts/fetch_data.py --region MISO --start 2022-01-01 --end 2023-12-31

    # Multiple specific regions
    python scripts/fetch_data.py --region PJM ERCO NYIS --start 2022-01-01 --end 2023-12-31

    # All enabled BAs (priority order, runs in sequence)
    python scripts/fetch_data.py --region ALL --start 2022-01-01 --end 2023-12-31

    # Priority 1 BAs only
    python scripts/fetch_data.py --region ALL --max-priority 1 --start 2022-01-01 --end 2023-12-31

    # Fetch only actual demand, not the EIA forecast
    python scripts/fetch_data.py --region MISO --types D --start 2022-01-01 --end 2023-12-31

    # Dry run (show what would be fetched, no API calls)
    python scripts/fetch_data.py --region MISO --dry-run

Environment:
    EIA_API_KEY — required; get a free key at https://www.eia.gov/opendata/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ── Make src/ importable when running from project root or scripts/ ─────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv()

from src.data.config_loader import get_ba_codes, get_ba_list, load_config
from src.data.eia_client import EIAClient, EIAClientError, save_raw

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fetch_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch EIA-930 hourly electricity demand data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--region", "-r",
        nargs="+",
        default=["MISO"],
        metavar="BA_CODE",
        help=(
            "One or more BA codes (e.g. MISO PJM ERCO), or 'ALL' to fetch every "
            "enabled Balancing Authority from config.yaml. Default: MISO"
        ),
    )
    parser.add_argument(
        "--start", "-s",
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date (inclusive). Default: value from config.yaml",
    )
    parser.add_argument(
        "--end", "-e",
        default=None,
        metavar="YYYY-MM-DD",
        help="End date (inclusive). Default: value from config.yaml",
    )
    parser.add_argument(
        "--types", "-t",
        nargs="+",
        default=None,
        choices=["D", "DF", "NG", "TI"],
        metavar="TYPE",
        help=(
            "EIA data types to fetch. D=actual demand, DF=EIA forecast. "
            "Default: D DF (from config.yaml)"
        ),
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        metavar="DIR",
        help="Directory to save raw CSVs. Default: data/raw from config.yaml",
    )
    parser.add_argument(
        "--max-priority",
        type=int,
        default=None,
        metavar="N",
        help="When --region ALL: only fetch BAs with priority <= N (1=most important)",
    )
    parser.add_argument(
        "--grid-region",
        choices=["eastern", "texas", "western"],
        default=None,
        help="When --region ALL: filter to one grid interconnection",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="EIA API key (overrides EIA_API_KEY env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without making API calls",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress per-page progress output",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config.yaml (auto-detected by default)",
    )
    return parser.parse_args(argv)


def resolve_regions(args: argparse.Namespace, cfg: dict) -> list[str]:
    """Resolve --region argument to a list of BA codes."""
    raw = [r.upper() for r in args.region]

    if raw == ["ALL"]:
        codes = get_ba_codes(
            cfg,
            enabled_only=True,
            region=args.grid_region,
            max_priority=args.max_priority,
        )
        return codes

    # Validate against config
    all_codes = {b["code"].upper() for b in cfg.get("balancing_authorities", [])}
    resolved = []
    for code in raw:
        if code not in all_codes:
            logger.warning(
                "BA code '%s' not found in config.yaml — including anyway", code
            )
        resolved.append(code)
    return resolved


def print_plan(regions: list[str], start: str, end: str, types: list[str], cfg: dict) -> None:
    """Print a summary of what will be fetched."""
    ba_map = {b["code"]: b for b in cfg.get("balancing_authorities", [])}
    print()
    print("=" * 60)
    print("  EIA-930 Fetch Plan")
    print("=" * 60)
    print(f"  Date range : {start}  -->  {end}")
    print(f"  Data types : {', '.join(types)}")
    print(f"  Regions    : {len(regions)} Balancing Authorit{'y' if len(regions)==1 else 'ies'}")
    print()
    for code in regions:
        info = ba_map.get(code, {})
        name = info.get("name", "")
        region = info.get("region", "")
        priority = info.get("priority", "?")
        print(f"    {code:<8} {region:<10} p={priority}  {name}")
    print()
    # Rough row estimate: ~17,520 rows per BA per 2 years × number of types
    est_rows_per_ba = 8760 * len(types)
    est_total = est_rows_per_ba * len(regions)
    print(f"  Est. rows  : ~{est_total:,} ({est_rows_per_ba:,} per BA × {len(regions)} BAs)")
    print("=" * 60)
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Load config
    cfg = load_config(args.config)

    # Resolve settings (CLI > config defaults)
    data_cfg = cfg.get("data", {})
    eia_cfg = cfg.get("eia", {})

    start = args.start or data_cfg.get("start_date", "2022-01-01")
    end = args.end or data_cfg.get("end_date", "2023-12-31")
    data_types = args.types or eia_cfg.get("data_types", ["D", "DF"])
    output_dir = args.output_dir or data_cfg.get("raw_data_dir", "data/raw")

    regions = resolve_regions(args, cfg)

    if not regions:
        print("ERROR: No Balancing Authorities selected. Check --region / --max-priority.")
        return 1

    print_plan(regions, start, end, data_types, cfg)

    if args.dry_run:
        print("DRY RUN — no API calls made.")
        return 0

    # Initialise client
    api_key = args.api_key or os.environ.get("EIA_API_KEY", "")
    if not api_key:
        print(
            "ERROR: No EIA_API_KEY found.\n"
            "  1. Register free at https://www.eia.gov/opendata/\n"
            "  2. Copy env.example to .env and add your key\n"
            "  3. Or pass --api-key YOUR_KEY\n"
        )
        return 1

    client = EIAClient(
        api_key=api_key,
        page_size=eia_cfg.get("page_size", 5000),
        rate_limit_delay=eia_cfg.get("rate_limit_delay", 0.5),
        timeout=eia_cfg.get("timeout", 30),
        retry_attempts=eia_cfg.get("retry_attempts", 3),
        retry_backoff=eia_cfg.get("retry_backoff", 2.0),
    )

    show_progress = not args.no_progress
    saved_files = []
    failed_regions = []

    for i, code in enumerate(regions, 1):
        print(f"\n[{i}/{len(regions)}] Fetching {code}  ({start} to {end})  types={data_types}")
        try:
            df = client.fetch_region(
                respondent=code,
                start=start,
                end=end,
                data_types=data_types,
                show_progress=show_progress,
            )

            if df.empty:
                print(f"  [WARNING] No data returned for {code}")
                failed_regions.append(code)
                continue

            fpath = save_raw(df, output_dir, code)
            saved_files.append(fpath)

            # Quick stats
            n_rows = len(df)
            n_d = len(df[df["type"] == "D"])
            n_df = len(df[df["type"] == "DF"])
            pct_missing = df["value_mwh"].isna().mean() * 100
            print(f"  Rows: {n_rows:,}  (D={n_d:,}, DF={n_df:,})  missing={pct_missing:.1f}%")
            print(f"  Saved: {fpath}")

        except EIAClientError as exc:
            logger.error("EIA API error for %s: %s", code, exc)
            print(f"  [ERROR] {exc}")
            failed_regions.append(code)
        except Exception as exc:
            logger.exception("Unexpected error for %s", code)
            print(f"  [ERROR] Unexpected: {exc}")
            failed_regions.append(code)

    # Summary
    print()
    print("=" * 60)
    print(f"  Fetch complete: {len(saved_files)} saved, {len(failed_regions)} failed")
    if saved_files:
        print(f"  Output dir: {Path(output_dir).resolve()}")
        for f in saved_files:
            print(f"    {Path(f).name}")
    if failed_regions:
        print(f"  Failed: {', '.join(failed_regions)}")
    print("=" * 60)

    return 0 if not failed_regions else 1


if __name__ == "__main__":
    sys.exit(main())
