#!/usr/bin/env python3
"""
fetch_bulk.py — Resumable bulk downloader for EIA-930 data.

Downloads all respondents (BAs + region aggregates) for one or more endpoints.
Saves per-BA CSVs to data/raw/{endpoint}/ and tracks progress in fetch_progress.json.
If interrupted, re-running resumes from where it left off.

Usage:
    # Terminal 1 — core demand data
    python scripts/fetch_bulk.py --endpoint region-data

    # Terminal 2 — exogenous features
    python scripts/fetch_bulk.py --endpoint fuel-type-data interchange-data region-sub-ba-data

    # Single BA test run
    python scripts/fetch_bulk.py --endpoint region-data --region MISO

    # Dry run (no API calls)
    python scripts/fetch_bulk.py --endpoint region-data --dry-run

    # Force re-download even if file exists
    python scripts/fetch_bulk.py --endpoint region-data --region MISO --force

    # Show current progress summary
    python scripts/fetch_bulk.py --status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Make src/ importable ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.data.config_loader import load_config
from src.data.eia_client import EIAClient, EIAClientError

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_bulk")

# ── Endpoint configuration ────────────────────────────────────────────────────
# Confirmed via live API: all data starts 2019-01-01T00
ENDPOINT_CONFIG = {
    "region-data": {
        "url_path": "region-data",
        "facet_key": "respondent",       # facets[respondent][] = BA code
        "data_types": ["D", "DF", "NG", "TI"],
        "extra_facets": lambda types: [("facets[type][]", t) for t in types],
        "start": "2019-01-01",
        "description": "Demand (D), EIA Forecast (DF), Net Generation (NG), Interchange (TI)",
        "est_rows_per_ba": 254760,
    },
    "fuel-type-data": {
        "url_path": "fuel-type-data",
        "facet_key": "respondent",
        "data_types": [],                # fetches all fuel types automatically
        "extra_facets": lambda types: [],
        "start": "2019-01-01",
        "description": "Hourly generation by fuel type (Coal, Gas, Wind, Solar, Hydro, Nuclear...)",
        "est_rows_per_ba": 456871,
    },
    "interchange-data": {
        "url_path": "interchange-data",
        "facet_key": "fromba",           # facets[fromba][] = source BA
        "data_types": [],
        "extra_facets": lambda types: [],
        "start": "2019-01-01",
        "description": "Hourly net power flows between neighboring BA pairs",
        "est_rows_per_ba": 624813,
    },
    "region-sub-ba-data": {
        "url_path": "region-sub-ba-data",
        "facet_key": "parent",           # facets[parent][] = parent BA code
        "data_types": [],
        "extra_facets": lambda types: [],
        "start": "2019-01-01",
        "description": "Hourly demand by subregion within each BA",
        "est_rows_per_ba": 381739,
    },
}

# ── Hardcoded respondent list (from live API, 81 total) ───────────────────────
ALL_RESPONDENTS = [
    # Individual BAs
    "AEC","AECI","AVA","AVRN","AZPS","BANC","BPAT","CHPD","CISO","CPLE","CPLW",
    "DEAA","DOPD","DUK","EEI","EPE","ERCO","FMPP","FPC","FPL","GCPD","GLHB",
    "GRID","GRIF","GVL","GWA","HGMA","HST","IID","IPCO","ISNE","JEA","LDWP",
    "LGEE","MISO","NEVP","NSB","NWMT","NYIS","PACE","PACW","PGE","PJM","PNM",
    "PSCO","PSEI","SC","SCL","SCEG","SEC","SEPA","SIKE","SOCO","SPA","SRP",
    "SWPP","TAL","TEC","TEPC","TIDC","TPWR","TVA","WACM","WALC","WAUW","WWA",
    "YAD",
    # Region aggregates (useful for national/regional benchmarks)
    "US48","MIDW","CAL","CAR","NW","SE","SW","TEX","NE","NY","TEN","CENT","FLA","MIDA",
]

PROGRESS_FILE = Path("data/raw/fetch_progress.json")


# ─────────────────────────────────────────────────────────────────────────────
# Progress tracking
# ─────────────────────────────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def mark_done(progress: dict, endpoint: str, ba: str, rows: int, fpath: str) -> None:
    if endpoint not in progress:
        progress[endpoint] = {}
    progress[endpoint][ba] = {
        "status": "done",
        "rows": rows,
        "file": fpath,
        "completed_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    save_progress(progress)


def mark_failed(progress: dict, endpoint: str, ba: str, error: str) -> None:
    if endpoint not in progress:
        progress[endpoint] = {}
    progress[endpoint][ba] = {
        "status": "failed",
        "error": error,
        "failed_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    save_progress(progress)


def is_done(progress: dict, endpoint: str, ba: str, force: bool) -> bool:
    if force:
        return False
    return progress.get(endpoint, {}).get(ba, {}).get("status") == "done"


# ─────────────────────────────────────────────────────────────────────────────
# Generic endpoint fetcher (handles different facet structures)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_one_ba(
    client: EIAClient,
    endpoint: str,
    ba: str,
    cfg: dict,
    eia_cfg: dict,
) -> tuple[list[dict], int]:
    """Fetch all pages for one BA from one endpoint. Returns (rows, total)."""
    ep_cfg = ENDPOINT_CONFIG[endpoint]
    start_period = f"{ep_cfg['start']}T00"
    end_period = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H")

    all_rows: list[dict] = []
    offset = 0
    page = 1
    total = None

    while True:
        query: list[tuple[str, str]] = [
            ("api_key", client.api_key),
            ("frequency", "hourly"),
            ("data[0]", "value"),
            (f"facets[{ep_cfg['facet_key']}][]", ba),
            ("start", start_period),
            ("end", end_period),
            ("sort[0][column]", "period"),
            ("sort[0][direction]", "asc"),
            ("length", str(client.page_size)),
            ("offset", str(offset)),
        ]
        # Add type facets where applicable
        for facet_tuple in ep_cfg["extra_facets"](ep_cfg["data_types"]):
            query.append(facet_tuple)

        url = f"https://api.eia.gov/v2/electricity/rto/{ep_cfg['url_path']}/data/"
        resp = client._session.get(url, params=query, timeout=client.timeout)

        if resp.status_code != 200:
            raise EIAClientError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        payload = resp.json()
        if "response" not in payload:
            error = payload.get("error", payload.get("message", str(payload)[:200]))
            raise EIAClientError(f"API error: {error}")

        body = payload["response"]
        data = body.get("data", [])
        if total is None:
            total = int(body.get("total", 0))

        all_rows.extend(data)

        fetched = len(all_rows)
        pct = (fetched / total * 100) if total > 0 else 0
        bar_len = 30
        filled = int(bar_len * fetched / total) if total > 0 else 0
        bar = "#" * filled + "-" * (bar_len - filled)
        print(
            f"\r  [{bar}] {fetched:>7,}/{total:>7,} ({pct:5.1f}%)  page {page}  ",
            end="",
            flush=True,
        )

        if fetched >= total or not data:
            break

        offset += client.page_size
        page += 1
        time.sleep(eia_cfg.get("rate_limit_delay", 0.3))

    print()  # newline after progress bar
    return all_rows, total or 0


# ─────────────────────────────────────────────────────────────────────────────
# Save raw rows to CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_endpoint_raw(rows: list[dict], endpoint: str, ba: str) -> Path:
    """Save fetched rows to data/raw/{endpoint}/{BA}_full_{endpoint}.csv"""
    import csv

    out_dir = Path("data/raw") / endpoint
    out_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    fname = f"{ba}_20190101_{today}_{endpoint}.csv"
    fpath = out_dir / fname

    if not rows:
        return fpath

    fieldnames = list(rows[0].keys())
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return fpath


# ─────────────────────────────────────────────────────────────────────────────
# Status printer
# ─────────────────────────────────────────────────────────────────────────────

def print_status() -> None:
    progress = load_progress()
    if not progress:
        print("No progress recorded yet.")
        return

    print()
    print("=" * 70)
    print("  EIA Bulk Download Progress")
    print("=" * 70)

    for endpoint, bas in progress.items():
        done = sum(1 for v in bas.values() if v.get("status") == "done")
        failed = sum(1 for v in bas.values() if v.get("status") == "failed")
        total_rows = sum(v.get("rows", 0) for v in bas.values() if v.get("status") == "done")
        print(f"\n  {endpoint}")
        print(f"    Done  : {done:>4}  |  Failed: {failed:>3}  |  Total rows: {total_rows:,}")
        if failed:
            fail_codes = [k for k, v in bas.items() if v.get("status") == "failed"]
            print(f"    Failed BAs: {', '.join(fail_codes)}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Bulk-download EIA-930 data for all Balancing Authorities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--endpoint", "-e",
        nargs="+",
        choices=list(ENDPOINT_CONFIG.keys()),
        default=["region-data"],
        help="Which endpoint(s) to download",
    )
    parser.add_argument(
        "--region", "-r",
        nargs="+",
        default=["ALL"],
        help="BA codes to fetch, or ALL (default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without making API calls",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if already completed",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show download progress summary and exit",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Seconds between paginated requests (default: 0.05)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.status:
        print_status()
        return 0

    api_key = os.environ.get("EIA_API_KEY", "")
    if not api_key:
        print("ERROR: EIA_API_KEY not set. Add it to .env")
        return 1

    cfg = load_config()
    eia_cfg = cfg.get("eia", {})
    eia_cfg["rate_limit_delay"] = args.delay

    # Resolve regions
    regions = args.region
    if regions == ["ALL"] or "ALL" in [r.upper() for r in regions]:
        regions = ALL_RESPONDENTS
    else:
        regions = [r.upper() for r in regions]

    endpoints = args.endpoint

    # Print plan
    print()
    print("=" * 70)
    print("  EIA-930 Bulk Download")
    print("=" * 70)
    for ep in endpoints:
        epc = ENDPOINT_CONFIG[ep]
        est_rows = epc["est_rows_per_ba"] * len(regions)
        est_pages = est_rows / 5000
        est_hours = est_pages * args.delay / 3600
        print(f"\n  Endpoint : {ep}")
        print(f"  Desc     : {epc['description']}")
        print(f"  BAs      : {len(regions)}")
        print(f"  Est rows : ~{est_rows:,}")
        print(f"  Est time : ~{est_hours:.1f}h  (at {args.delay}s/request)")
        print(f"  Output   : data/raw/{ep}/")

    print()

    if args.dry_run:
        print("DRY RUN — no API calls made.")
        print(f"\nRegions that would be fetched ({len(regions)}):")
        for r in regions:
            print(f"  {r}")
        return 0

    client = EIAClient(
        api_key=api_key,
        page_size=5000,
        rate_limit_delay=args.delay,
        timeout=eia_cfg.get("timeout", 30),
        retry_attempts=eia_cfg.get("retry_attempts", 3),
        retry_backoff=eia_cfg.get("retry_backoff", 2.0),
    )

    progress = load_progress()
    grand_total_rows = 0
    grand_failed = []

    for endpoint in endpoints:
        epc = ENDPOINT_CONFIG[endpoint]
        skipped = 0
        done_count = 0
        failed_count = 0
        endpoint_rows = 0

        print(f"\n{'='*70}")
        print(f"  ENDPOINT: {endpoint.upper()}")
        print(f"  {epc['description']}")
        print(f"{'='*70}")

        for i, ba in enumerate(regions, 1):
            prefix = f"[{i:>3}/{len(regions)}] {ba:<10}"

            if is_done(progress, endpoint, ba, args.force):
                cached = progress[endpoint][ba]
                skipped += 1
                print(f"{prefix} SKIP (already done — {cached.get('rows',0):,} rows)")
                endpoint_rows += cached.get("rows", 0)
                continue

            print(f"{prefix} fetching...", flush=True)
            t0 = time.time()

            try:
                rows, total = fetch_one_ba(client, endpoint, ba, cfg, eia_cfg)

                if not rows:
                    print(f"{prefix} NO DATA — this BA has no {endpoint} data (skipping)")
                    if endpoint not in progress:
                        progress[endpoint] = {}
                    progress[endpoint][ba] = {"status": "no_data"}
                    save_progress(progress)
                    continue

                fpath = save_endpoint_raw(rows, endpoint, ba)
                elapsed = time.time() - t0
                mark_done(progress, endpoint, ba, len(rows), str(fpath))
                done_count += 1
                endpoint_rows += len(rows)
                grand_total_rows += len(rows)

                print(
                    f"{prefix} OK  {len(rows):>8,} rows  "
                    f"saved: {fpath.name}  ({elapsed:.0f}s)"
                )

            except EIAClientError as exc:
                elapsed = time.time() - t0
                print(f"{prefix} ERROR — {exc}  ({elapsed:.0f}s)")
                mark_failed(progress, endpoint, ba, str(exc))
                failed_count += 1
                grand_failed.append(f"{endpoint}/{ba}")

            except Exception as exc:
                elapsed = time.time() - t0
                print(f"{prefix} UNEXPECTED ERROR — {exc}  ({elapsed:.0f}s)")
                mark_failed(progress, endpoint, ba, str(exc))
                failed_count += 1
                grand_failed.append(f"{endpoint}/{ba}")

        print(f"\n  {endpoint} summary:")
        print(f"    Done={done_count}  Skipped={skipped}  Failed={failed_count}")
        print(f"    Total rows this endpoint: {endpoint_rows:,}")

    # Final summary
    print()
    print("=" * 70)
    print("  BULK DOWNLOAD COMPLETE")
    print(f"  Total rows saved : {grand_total_rows:,}")
    print(f"  Failed           : {len(grand_failed)}")
    if grand_failed:
        print(f"  Failed items     : {', '.join(grand_failed)}")
    print(f"  Progress file    : {PROGRESS_FILE}")
    print("  Re-run to retry failed items (completed ones will be skipped)")
    print("=" * 70)

    return 0 if not grand_failed else 1


if __name__ == "__main__":
    sys.exit(main())
