"""
eia_client.py — EIA Open Data API v2 client for EIA-930 electricity demand data.

Fetches hourly Balancing Authority data including:
  D  — Actual demand (MWh)
  DF — EIA's own demand forecast (MWh)  ← the benchmark we aim to beat

EIA API v2 docs: https://www.eia.gov/opendata/documentation.php
Register for a free key: https://www.eia.gov/opendata/

Usage:
    from src.data.eia_client import EIAClient

    client = EIAClient(api_key="your_key")
    df = client.fetch_region(
        respondent="MISO",
        start="2022-01-01",
        end="2023-12-31",
        data_types=["D", "DF"],
    )
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# EIA API v2 base endpoint for RTO/BA hourly data
_BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
_PAGE_SIZE = 5000  # EIA hard cap per request


class EIAClientError(Exception):
    """Raised when the EIA API returns an error or unexpected response."""


class EIAClient:
    """Paginated client for the EIA Open Data API v2.

    Args:
        api_key: EIA API key. If None, reads EIA_API_KEY from environment.
        base_url: Override the default API endpoint.
        page_size: Rows per paginated request (max 5000).
        rate_limit_delay: Seconds to wait between paginated requests.
        timeout: HTTP request timeout in seconds.
        retry_attempts: Number of times to retry a failed request.
        retry_backoff: Exponential backoff multiplier between retries.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = _BASE_URL,
        page_size: int = _PAGE_SIZE,
        rate_limit_delay: float = 0.5,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("EIA_API_KEY", "")
        if not self.api_key:
            raise EIAClientError(
                "No EIA API key found. Set EIA_API_KEY environment variable "
                "or pass api_key= to EIAClient()."
            )
        self.base_url = base_url
        self.page_size = min(page_size, _PAGE_SIZE)
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout

        # Build a session with retry logic
        self._session = requests.Session()
        retry = Retry(
            total=retry_attempts,
            backoff_factor=retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def fetch_region(
        self,
        respondent: str,
        start: str,
        end: str,
        data_types: list[str] | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Fetch all hourly rows for one Balancing Authority between start and end.

        Args:
            respondent: BA code, e.g. "MISO", "PJM", "ERCO".
            start: ISO date string "YYYY-MM-DD" (inclusive).
            end: ISO date string "YYYY-MM-DD" (inclusive).
            data_types: List of EIA type codes to fetch. Defaults to ["D", "DF"].
                D  = actual hourly demand
                DF = EIA's own demand forecast
            show_progress: Print page-fetch progress to stdout.

        Returns:
            DataFrame with columns:
                period (datetime, UTC-aware), respondent, type, type_name,
                value_mwh (float), fetched_at (datetime)
        """
        if data_types is None:
            data_types = ["D", "DF"]

        # Convert dates to EIA period format: YYYY-MM-DDTHH
        start_period = _date_to_period(start, hour=0)
        end_period = _date_to_period(end, hour=23)

        logger.info(
            "Fetching %s | %s | %s → %s", respondent, data_types, start_period, end_period
        )

        all_rows: list[dict] = []
        offset = 0
        page = 1

        while True:
            params = self._build_params(
                respondent=respondent,
                data_types=data_types,
                start=start_period,
                end=end_period,
                offset=offset,
            )

            if show_progress:
                print(
                    f"  [{respondent}] page {page} (offset={offset}) ...",
                    end="\r",
                    flush=True,
                )

            data, total = self._get_page(params)
            all_rows.extend(data)

            fetched = len(all_rows)
            if show_progress:
                print(
                    f"  [{respondent}] fetched {fetched:,} / {total:,} rows     ",
                    end="\r",
                    flush=True,
                )

            if fetched >= total or not data:
                break

            offset += self.page_size
            page += 1
            time.sleep(self.rate_limit_delay)

        if show_progress:
            print()  # newline after \r progress

        if not all_rows:
            logger.warning("No data returned for %s %s→%s", respondent, start, end)
            return pd.DataFrame()

        df = self._normalize(all_rows)
        logger.info(
            "Done: %s rows for %s (%s → %s)",
            len(df),
            respondent,
            df["period"].min(),
            df["period"].max(),
        )
        return df

    def fetch_multiple_regions(
        self,
        respondents: list[str],
        start: str,
        end: str,
        data_types: list[str] | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Fetch data for multiple BAs and concatenate into one DataFrame.

        Args:
            respondents: List of BA codes, e.g. ["MISO", "PJM", "ERCO"].
            start: ISO date string "YYYY-MM-DD".
            end: ISO date string "YYYY-MM-DD".
            data_types: EIA type codes to fetch. Defaults to ["D", "DF"].
            show_progress: Print progress.

        Returns:
            Combined DataFrame for all BAs (same schema as fetch_region).
        """
        frames: list[pd.DataFrame] = []
        total = len(respondents)

        for i, code in enumerate(respondents, 1):
            print(f"\n[{i}/{total}] Fetching {code}...")
            try:
                df = self.fetch_region(
                    respondent=code,
                    start=start,
                    end=end,
                    data_types=data_types,
                    show_progress=show_progress,
                )
                if not df.empty:
                    frames.append(df)
                    print(f"  [{code}] OK — {len(df):,} rows")
                else:
                    print(f"  [{code}] WARNING — no data returned")
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", code, exc)
                print(f"  [{code}] ERROR — {exc}")

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_params(
        self,
        respondent: str,
        data_types: list[str],
        start: str,
        end: str,
        offset: int,
    ) -> dict[str, Any]:
        """Build EIA API v2 query parameters."""
        # EIA v2 uses multi-value params: facets[type][]=D&facets[type][]=DF
        params: dict[str, Any] = {
            "api_key": self.api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": respondent,
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": self.page_size,
            "offset": offset,
        }
        # EIA API v2 requires repeated keys for multi-value facets
        # We'll handle this in _get_page using the `params` list format
        return {
            "_respondent": respondent,
            "_data_types": data_types,
            "_start": start,
            "_end": end,
            "_offset": offset,
        }

    def _get_page(self, params: dict) -> tuple[list[dict], int]:
        """Execute one paginated GET request and return (rows, total_count)."""
        respondent = params["_respondent"]
        data_types = params["_data_types"]

        # Build query string manually for multi-value facets
        # requests library needs a list of tuples for repeated keys
        query: list[tuple[str, str]] = [
            ("api_key", self.api_key),
            ("frequency", "hourly"),
            ("data[0]", "value"),
            ("facets[respondent][]", respondent),
            ("start", params["_start"]),
            ("end", params["_end"]),
            ("sort[0][column]", "period"),
            ("sort[0][direction]", "asc"),
            ("length", str(self.page_size)),
            ("offset", str(params["_offset"])),
        ]
        for dtype in data_types:
            query.append(("facets[type][]", dtype))

        resp = self._session.get(
            self.base_url,
            params=query,
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            raise EIAClientError(
                f"EIA API returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise EIAClientError(f"Failed to parse EIA response as JSON: {exc}") from exc

        if "response" not in payload:
            # EIA sometimes returns {"error": "..."} for bad API keys
            error_msg = payload.get("error", payload.get("message", str(payload)[:200]))
            raise EIAClientError(f"EIA API error: {error_msg}")

        response_body = payload["response"]
        data = response_body.get("data", [])
        total = int(response_body.get("total", len(data)))

        return data, total

    @staticmethod
    def _normalize(rows: list[dict]) -> pd.DataFrame:
        """Convert raw API rows to a clean DataFrame."""
        fetched_at = datetime.now(tz=timezone.utc)
        records = []
        for row in rows:
            period_str = row.get("period", "")
            try:
                # EIA period format: "2022-01-01T00" (local time, but we treat as UTC)
                period = pd.to_datetime(period_str, format="%Y-%m-%dT%H", utc=True)
            except (ValueError, TypeError):
                logger.warning("Could not parse period '%s', skipping row", period_str)
                continue

            value_raw = row.get("value")
            try:
                value = float(value_raw) if value_raw is not None else float("nan")
            except (ValueError, TypeError):
                value = float("nan")

            records.append(
                {
                    "period": period,
                    "respondent": row.get("respondent", ""),
                    "type": row.get("type", ""),
                    "type_name": row.get("type-name", ""),
                    "value_mwh": value,
                    "fetched_at": fetched_at,
                }
            )

        df = pd.DataFrame(records)
        if df.empty:
            return df

        df = df.sort_values(["respondent", "type", "period"]).reset_index(drop=True)
        return df


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────


def _date_to_period(date_str: str, hour: int = 0) -> str:
    """Convert 'YYYY-MM-DD' to EIA period format 'YYYY-MM-DDTHH'."""
    date_str = date_str.strip()
    if "T" in date_str:
        return date_str  # already in EIA format
    return f"{date_str}T{hour:02d}"


def save_raw(df: pd.DataFrame, output_dir: str | Path, respondent: str) -> Path:
    """Save a raw fetched DataFrame to CSV in the data/raw directory.

    File naming: {respondent}_{start}_{end}_raw.csv

    Args:
        df: DataFrame as returned by EIAClient.fetch_region()
        output_dir: Directory to write to (e.g. "data/raw")
        respondent: BA code used to name the file

    Returns:
        Path to the saved CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        raise ValueError(f"Cannot save empty DataFrame for {respondent}")

    start = df["period"].min().strftime("%Y%m%d")
    end = df["period"].max().strftime("%Y%m%d")
    fname = f"{respondent}_{start}_{end}_raw.csv"
    fpath = output_dir / fname

    df.to_csv(fpath, index=False)
    logger.info("Saved %d rows to %s", len(df), fpath)
    return fpath


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Load a previously saved raw CSV back into a DataFrame.

    Parses the 'period' and 'fetched_at' columns as datetimes.
    """
    df = pd.read_csv(csv_path, parse_dates=["period", "fetched_at"])
    if "period" in df.columns:
        df["period"] = pd.to_datetime(df["period"], utc=True)
    return df
