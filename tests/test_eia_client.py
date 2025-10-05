"""
tests/test_eia_client.py — Unit tests for EIAClient.

Tests use unittest.mock to simulate API responses — no real API key needed.
Run with: pytest tests/test_eia_client.py -v
"""

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Make src/ importable when running from project root ─────────────────────
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.eia_client import (
    EIAClient,
    EIAClientError,
    _date_to_period,
    load_raw,
    save_raw,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures & helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_eia_response(
    rows: list[dict],
    total: int | None = None,
) -> dict:
    """Build a mock EIA API v2 response payload."""
    return {
        "response": {
            "total": str(total if total is not None else len(rows)),
            "dateFormat": "YYYY-MM-DDTHH",
            "frequency": "hourly",
            "data": rows,
        }
    }


def _sample_rows(
    respondent: str = "MISO",
    dtype: str = "D",
    n: int = 5,
    start_hour: int = 0,
) -> list[dict]:
    """Generate n fake EIA hourly rows."""
    return [
        {
            "period": f"2022-01-01T{(start_hour + i) % 24:02d}",
            "respondent": respondent,
            "type": dtype,
            "type-name": "Demand" if dtype == "D" else "Demand Forecast",
            "value": str(80000 + i * 100),
            "value-units": "megawatthours",
        }
        for i in range(n)
    ]


@pytest.fixture()
def client():
    """EIAClient with a dummy API key (no real calls made in tests)."""
    return EIAClient(api_key="TEST_KEY_DUMMY")


# ──────────────────────────────────────────────────────────────────────────────
# _date_to_period
# ──────────────────────────────────────────────────────────────────────────────


class TestDateToPeriod:
    def test_basic_conversion(self):
        assert _date_to_period("2022-01-01") == "2022-01-01T00"

    def test_custom_hour(self):
        assert _date_to_period("2022-01-01", hour=23) == "2022-01-01T23"

    def test_already_period_format(self):
        assert _date_to_period("2022-06-15T12") == "2022-06-15T12"

    def test_strips_whitespace(self):
        assert _date_to_period("  2022-03-10  ") == "2022-03-10T00"


# ──────────────────────────────────────────────────────────────────────────────
# EIAClient initialisation
# ──────────────────────────────────────────────────────────────────────────────


class TestEIAClientInit:
    def test_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("EIA_API_KEY", raising=False)
        with pytest.raises(EIAClientError, match="No EIA API key"):
            EIAClient()

    def test_reads_key_from_env(self, monkeypatch):
        monkeypatch.setenv("EIA_API_KEY", "env_key_123")
        c = EIAClient()
        assert c.api_key == "env_key_123"

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("EIA_API_KEY", "env_key")
        c = EIAClient(api_key="explicit_key")
        assert c.api_key == "explicit_key"

    def test_page_size_capped_at_5000(self):
        c = EIAClient(api_key="k", page_size=99999)
        assert c.page_size == 5000

    def test_custom_page_size(self):
        c = EIAClient(api_key="k", page_size=1000)
        assert c.page_size == 1000


# ──────────────────────────────────────────────────────────────────────────────
# Single-page fetch
# ──────────────────────────────────────────────────────────────────────────────


class TestFetchRegionSinglePage:
    def _mock_response(self, payload: dict, status: int = 200) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.json.return_value = payload
        mock_resp.text = json.dumps(payload)
        return mock_resp

    def test_returns_dataframe_with_correct_columns(self, client):
        rows = _sample_rows("MISO", "D", n=3) + _sample_rows("MISO", "DF", n=3)
        payload = _make_eia_response(rows, total=6)

        with patch.object(client._session, "get", return_value=self._mock_response(payload)):
            df = client.fetch_region("MISO", "2022-01-01", "2022-01-01", show_progress=False)

        assert not df.empty
        assert set(df.columns) >= {"period", "respondent", "type", "type_name", "value_mwh"}

    def test_period_column_is_utc_datetime(self, client):
        rows = _sample_rows("MISO", "D", n=2)
        payload = _make_eia_response(rows)

        with patch.object(client._session, "get", return_value=self._mock_response(payload)):
            df = client.fetch_region("MISO", "2022-01-01", "2022-01-01", show_progress=False)

        assert pd.api.types.is_datetime64_any_dtype(df["period"])
        assert df["period"].dt.tz == timezone.utc

    def test_value_mwh_is_float(self, client):
        rows = _sample_rows("MISO", "D", n=3)
        payload = _make_eia_response(rows)

        with patch.object(client._session, "get", return_value=self._mock_response(payload)):
            df = client.fetch_region("MISO", "2022-01-01", "2022-01-01", show_progress=False)

        assert df["value_mwh"].dtype == float

    def test_both_types_returned(self, client):
        rows = _sample_rows("ERCO", "D", n=5) + _sample_rows("ERCO", "DF", n=5)
        payload = _make_eia_response(rows, total=10)

        with patch.object(client._session, "get", return_value=self._mock_response(payload)):
            df = client.fetch_region(
                "ERCO", "2022-01-01", "2022-01-01",
                data_types=["D", "DF"], show_progress=False,
            )

        assert set(df["type"].unique()) == {"D", "DF"}
        assert len(df[df["type"] == "D"]) == 5
        assert len(df[df["type"] == "DF"]) == 5

    def test_empty_response_returns_empty_df(self, client):
        payload = _make_eia_response([], total=0)

        with patch.object(client._session, "get", return_value=self._mock_response(payload)):
            df = client.fetch_region("MISO", "2022-01-01", "2022-01-01", show_progress=False)

        assert df.empty

    def test_http_error_raises_client_error(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        with patch.object(client._session, "get", return_value=mock_resp):
            with pytest.raises(EIAClientError, match="HTTP 403"):
                client.fetch_region("MISO", "2022-01-01", "2022-01-01", show_progress=False)

    def test_api_error_in_payload_raises(self, client):
        payload = {"error": "Invalid API key"}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload
        mock_resp.text = json.dumps(payload)

        with patch.object(client._session, "get", return_value=mock_resp):
            with pytest.raises(EIAClientError, match="Invalid API key"):
                client.fetch_region("MISO", "2022-01-01", "2022-01-01", show_progress=False)


# ──────────────────────────────────────────────────────────────────────────────
# Pagination
# ──────────────────────────────────────────────────────────────────────────────


class TestPagination:
    def test_fetches_all_pages(self, client):
        """When total > page_size, client should make multiple requests."""
        page1_rows = _sample_rows("PJM", "D", n=3)
        page2_rows = _sample_rows("PJM", "D", n=2, start_hour=3)

        def mock_get(url, params, timeout):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = ""
            # Check offset param to determine which page
            params_dict = dict(params)
            offset = int(params_dict.get("offset", 0))
            if offset == 0:
                mock_resp.json.return_value = _make_eia_response(page1_rows, total=5)
            else:
                mock_resp.json.return_value = _make_eia_response(page2_rows, total=5)
            return mock_resp

        small_client = EIAClient(api_key="TEST_KEY", page_size=3)
        with patch.object(small_client._session, "get", side_effect=mock_get):
            df = small_client.fetch_region(
                "PJM", "2022-01-01", "2022-01-01", data_types=["D"], show_progress=False
            )

        assert len(df) == 5

    def test_stops_when_no_more_data(self, client):
        """If API returns 0 rows on second page, stop paginating."""
        page1_rows = _sample_rows("TVA", "D", n=3)

        call_count = 0

        def mock_get(url, params, timeout):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = ""
            # First call returns data, subsequent calls return empty
            if call_count == 1:
                mock_resp.json.return_value = _make_eia_response(page1_rows, total=10)
            else:
                mock_resp.json.return_value = _make_eia_response([], total=10)
            return mock_resp

        small_client = EIAClient(api_key="TEST_KEY", page_size=3)
        with patch.object(small_client._session, "get", side_effect=mock_get):
            df = small_client.fetch_region(
                "TVA", "2022-01-01", "2022-01-01", data_types=["D"], show_progress=False
            )

        assert len(df) == 3
        assert call_count == 2


# ──────────────────────────────────────────────────────────────────────────────
# Multi-region fetch
# ──────────────────────────────────────────────────────────────────────────────


class TestFetchMultipleRegions:
    def test_combines_all_regions(self, client):
        miso_rows = _sample_rows("MISO", "D", n=4)
        pjm_rows = _sample_rows("PJM", "D", n=3)

        responses = {
            "MISO": _make_eia_response(miso_rows, total=4),
            "PJM": _make_eia_response(pjm_rows, total=3),
        }

        call_respondents = []

        def mock_get(url, params, timeout):
            params_dict = dict(params)
            resp_code = params_dict.get("facets[respondent][]", "")
            call_respondents.append(resp_code)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = responses[resp_code]
            mock_resp.text = ""
            return mock_resp

        with patch.object(client._session, "get", side_effect=mock_get):
            df = client.fetch_multiple_regions(
                ["MISO", "PJM"],
                start="2022-01-01",
                end="2022-01-01",
                data_types=["D"],
                show_progress=False,
            )

        assert len(df) == 7
        assert set(df["respondent"].unique()) == {"MISO", "PJM"}

    def test_skips_failed_region_continues(self, client):
        good_rows = _sample_rows("ERCO", "D", n=3)

        def mock_get(url, params, timeout):
            params_dict = dict(params)
            resp_code = params_dict.get("facets[respondent][]", "")
            mock_resp = MagicMock()
            if resp_code == "BAD_CODE":
                mock_resp.status_code = 400
                mock_resp.text = "Bad request"
                mock_resp.json.return_value = {"error": "unknown respondent"}
            else:
                mock_resp.status_code = 200
                mock_resp.json.return_value = _make_eia_response(good_rows)
                mock_resp.text = ""
            return mock_resp

        with patch.object(client._session, "get", side_effect=mock_get):
            df = client.fetch_multiple_regions(
                ["BAD_CODE", "ERCO"],
                start="2022-01-01",
                end="2022-01-01",
                data_types=["D"],
                show_progress=False,
            )

        # Should still have ERCO data despite BAD_CODE failing
        assert len(df) == 3
        assert df["respondent"].iloc[0] == "ERCO"


# ──────────────────────────────────────────────────────────────────────────────
# save_raw / load_raw
# ──────────────────────────────────────────────────────────────────────────────


class TestSaveLoadRaw:
    def _sample_df(self, respondent: str = "MISO") -> pd.DataFrame:
        rows = _sample_rows(respondent, "D", n=5) + _sample_rows(respondent, "DF", n=5)
        from src.data.eia_client import EIAClient
        # Use _normalize directly to get a properly typed DataFrame
        return EIAClient._normalize(rows)

    def test_save_creates_csv_file(self, tmp_path):
        df = self._sample_df("MISO")
        fpath = save_raw(df, tmp_path, "MISO")
        assert fpath.exists()
        assert fpath.suffix == ".csv"
        assert "MISO" in fpath.name

    def test_save_filename_contains_dates(self, tmp_path):
        df = self._sample_df("PJM")
        fpath = save_raw(df, tmp_path, "PJM")
        assert "20220101" in fpath.name

    def test_roundtrip_preserves_row_count(self, tmp_path):
        df = self._sample_df("ERCO")
        fpath = save_raw(df, tmp_path, "ERCO")
        df2 = load_raw(fpath)
        assert len(df2) == len(df)

    def test_roundtrip_preserves_value_mwh(self, tmp_path):
        df = self._sample_df("TVA")
        fpath = save_raw(df, tmp_path, "TVA")
        df2 = load_raw(fpath)
        assert list(df["value_mwh"]) == pytest.approx(list(df2["value_mwh"]))

    def test_save_raises_on_empty_df(self, tmp_path):
        with pytest.raises(ValueError, match="empty DataFrame"):
            save_raw(pd.DataFrame(), tmp_path, "MISO")

    def test_save_creates_output_dir_if_missing(self, tmp_path):
        new_dir = tmp_path / "deeply" / "nested" / "raw"
        df = self._sample_df("MISO")
        fpath = save_raw(df, new_dir, "MISO")
        assert new_dir.exists()
        assert fpath.exists()
