"""
tests/test_ingest.py — Tests for the ingestion pipeline.

Tests use both synthetic data and (when available) the real MISO CSV
in data/raw/region-data/ to validate the full pipeline end-to-end.

Run with: pytest tests/test_ingest.py -v
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.data.database import (
    FuelTypeData, InterchangeData, RegionData, SubBaData,
    create_all_tables, get_engine, get_table_counts,
)
from src.data.ingest import (
    _detect_endpoint, _limited_interpolate, _process_region_data,
    ingest_file, load_region_series, load_wide,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def db_engine(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path}/test.db")
    create_all_tables(engine)
    return engine


def _make_region_csv(tmp_path: Path, ba: str = "TEST", n_hours: int = 200) -> Path:
    """Generate a synthetic region-data CSV with D and DF series."""
    periods = pd.date_range("2022-01-01", periods=n_hours, freq="h", tz="UTC")
    rows = []
    for period in periods:
        for dtype, tname, base in [("D", "Demand", 80000), ("DF", "Day-ahead demand forecast", 81000)]:
            rows.append({
                "period": period.strftime("%Y-%m-%dT%H"),
                "respondent": ba,
                "respondent-name": f"{ba} Test BA",
                "type": dtype,
                "type-name": tname,
                "value": base + np.random.randint(-5000, 5000),
                "value-units": "megawatthours",
            })
    df = pd.DataFrame(rows)
    fpath = tmp_path / f"{ba}_20220101_20220110_region-data.csv"
    df.to_csv(fpath, index=False)
    return fpath


def _make_region_csv_with_gaps(tmp_path: Path, gap_start: int = 10, gap_len: int = 3) -> Path:
    """Synthetic CSV with a deliberate gap in the D series."""
    periods = pd.date_range("2022-01-01", periods=100, freq="h", tz="UTC")
    rows = []
    for i, period in enumerate(periods):
        for dtype in ("D", "DF"):
            in_gap = (dtype == "D") and (gap_start <= i < gap_start + gap_len)
            rows.append({
                "period": period.strftime("%Y-%m-%dT%H"),
                "respondent": "GAP",
                "respondent-name": "Gap Test BA",
                "type": dtype,
                "type-name": dtype,
                "value": "" if in_gap else 80000 + i * 10,
                "value-units": "megawatthours",
            })
    df = pd.DataFrame(rows)
    fpath = tmp_path / "GAP_20220101_20220105_region-data.csv"
    df.to_csv(fpath, index=False)
    return fpath


def _make_region_csv_with_spike(tmp_path: Path) -> Path:
    """Synthetic CSV with one extreme anomalous spike."""
    n = 300
    periods = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    rows = []
    for i, period in enumerate(periods):
        value = 80000  # flat baseline — spike stands out clearly
        if i == 150:
            value = 999_999  # extreme spike
        rows.append({
            "period": period.strftime("%Y-%m-%dT%H"),
            "respondent": "SPIKE",
            "respondent-name": "Spike BA",
            "type": "D",
            "type-name": "Demand",
            "value": value,
            "value-units": "megawatthours",
        })
    df = pd.DataFrame(rows)
    fpath = tmp_path / "SPIKE_20220101_20220114_region-data.csv"
    df.to_csv(fpath, index=False)
    return fpath


def _make_fuel_csv(tmp_path: Path, ba: str = "TEST") -> Path:
    periods = pd.date_range("2022-01-01", periods=50, freq="h", tz="UTC")
    rows = []
    for period in periods:
        for fuel in ("COL", "NG", "WND", "SUN"):
            rows.append({
                "period": period.strftime("%Y-%m-%dT%H"),
                "respondent": ba,
                "respondent-name": f"{ba} Test",
                "fueltype": fuel,
                "type-name": fuel,
                "value": np.random.randint(0, 10000),
                "value-units": "megawatthours",
            })
    df = pd.DataFrame(rows)
    fpath = tmp_path / f"{ba}_20220101_20220103_fuel-type-data.csv"
    df.to_csv(fpath, index=False)
    return fpath


def _make_interchange_csv(tmp_path: Path) -> Path:
    periods = pd.date_range("2022-01-01", periods=50, freq="h", tz="UTC")
    rows = [
        {
            "period": p.strftime("%Y-%m-%dT%H"),
            "fromba": "MISO",
            "fromba-name": "MISO",
            "toba": "PJM",
            "toba-name": "PJM",
            "value": np.random.randint(-5000, 5000),
            "value-units": "megawatthours",
        }
        for p in periods
    ]
    df = pd.DataFrame(rows)
    fpath = tmp_path / "MISO_20220101_20220103_interchange-data.csv"
    df.to_csv(fpath, index=False)
    return fpath


def _make_subba_csv(tmp_path: Path) -> Path:
    periods = pd.date_range("2022-01-01", periods=50, freq="h", tz="UTC")
    rows = [
        {
            "period": p.strftime("%Y-%m-%dT%H"),
            "subba": "MISO_NORTH",
            "subba-name": "MISO North",
            "parent": "MISO",
            "value": np.random.randint(10000, 30000),
            "value-units": "megawatthours",
        }
        for p in periods
    ]
    df = pd.DataFrame(rows)
    fpath = tmp_path / "MISO_20220101_20220103_region-sub-ba-data.csv"
    df.to_csv(fpath, index=False)
    return fpath


# ─────────────────────────────────────────────────────────────────────────────
# _detect_endpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectEndpoint:
    def test_region_data(self, tmp_path):
        f = tmp_path / "MISO_20220101_20231231_region-data.csv"
        f.touch()
        assert _detect_endpoint(f) == "region-data"

    def test_fuel_type(self, tmp_path):
        f = tmp_path / "PJM_20190101_20260409_fuel-type-data.csv"
        f.touch()
        assert _detect_endpoint(f) == "fuel-type-data"

    def test_interchange(self, tmp_path):
        f = tmp_path / "ERCO_20190101_interchange-data.csv"
        f.touch()
        assert _detect_endpoint(f) == "interchange-data"

    def test_sub_ba(self, tmp_path):
        f = tmp_path / "MISO_region-sub-ba-data.csv"
        f.touch()
        assert _detect_endpoint(f) == "region-sub-ba-data"

    def test_unknown_raises(self, tmp_path):
        f = tmp_path / "unknown_file.csv"
        f.touch()
        with pytest.raises(ValueError, match="Cannot detect"):
            _detect_endpoint(f)


# ─────────────────────────────────────────────────────────────────────────────
# _limited_interpolate
# ─────────────────────────────────────────────────────────────────────────────

class TestLimitedInterpolate:
    def _series(self, values):
        idx = pd.date_range("2022-01-01", periods=len(values), freq="h", tz="UTC")
        return pd.Series(values, index=idx, dtype=float)

    def test_no_gaps_unchanged(self):
        s = self._series([1.0, 2.0, 3.0, 4.0])
        out = _limited_interpolate(s, max_gap=3)
        pd.testing.assert_series_equal(s, out)

    def test_single_gap_filled(self):
        s = self._series([10.0, np.nan, 30.0])
        out = _limited_interpolate(s, max_gap=3)
        assert out.iloc[1] == pytest.approx(20.0)

    def test_gap_within_limit_filled(self):
        s = self._series([0.0, np.nan, np.nan, np.nan, 4.0])
        out = _limited_interpolate(s, max_gap=3)
        assert out.isna().sum() == 0

    def test_gap_exceeding_limit_not_filled(self):
        # 4 NaNs exceeds max_gap=3
        s = self._series([0.0, np.nan, np.nan, np.nan, np.nan, 5.0])
        out = _limited_interpolate(s, max_gap=3)
        assert out.isna().sum() == 4

    def test_mixed_gaps(self):
        # gap of 2 (ok) and gap of 5 (too large)
        vals = [1.0, np.nan, np.nan, 4.0] + [5.0] + [np.nan]*5 + [11.0]
        s = self._series(vals)
        out = _limited_interpolate(s, max_gap=3)
        # First gap (len=2) should be filled
        assert pd.notna(out.iloc[1]) and pd.notna(out.iloc[2])
        # Second gap (len=5) should still be NaN
        assert out.iloc[5:10].isna().all()


# ─────────────────────────────────────────────────────────────────────────────
# _process_region_data
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessRegionData:
    def test_output_has_required_columns(self, tmp_path):
        fpath = _make_region_csv(tmp_path, n_hours=100)
        from src.data.ingest import _read_raw
        df_raw = _read_raw(fpath, "region-data")
        df_clean, stats = _process_region_data(df_raw)
        required = {"period", "respondent", "type", "value_mwh", "is_imputed", "is_anomaly"}
        assert required.issubset(set(df_clean.columns))

    def test_gap_filling_marks_imputed(self, tmp_path):
        fpath = _make_region_csv_with_gaps(tmp_path, gap_start=10, gap_len=3)
        from src.data.ingest import _read_raw
        df_raw = _read_raw(fpath, "region-data")
        df_clean, stats = _process_region_data(df_raw)
        assert stats["imputed"] == 3
        imputed_rows = df_clean[(df_clean["is_imputed"] == 1) & (df_clean["type"] == "D")]
        assert len(imputed_rows) == 3

    def test_anomaly_detection_flags_spike(self, tmp_path):
        fpath = _make_region_csv_with_spike(tmp_path)
        from src.data.ingest import _read_raw
        df_raw = _read_raw(fpath, "region-data")
        df_clean, stats = _process_region_data(df_raw)
        assert stats["anomaly"] >= 1

    def test_no_duplicate_periods_per_type(self, tmp_path):
        fpath = _make_region_csv(tmp_path, n_hours=200)
        from src.data.ingest import _read_raw
        df_raw = _read_raw(fpath, "region-data")
        df_clean, _ = _process_region_data(df_raw)
        dupes = df_clean.duplicated(subset=["period", "respondent", "type"])
        assert not dupes.any()


# ─────────────────────────────────────────────────────────────────────────────
# ingest_file — region-data
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestFileRegionData:
    def test_successful_ingest(self, tmp_path, db_engine):
        fpath = _make_region_csv(tmp_path, n_hours=100)
        result = ingest_file(fpath, engine=db_engine, verbose=False)
        assert result["status"] == "success"
        assert result["rows_ingested"] > 0

    def test_rows_match_expected(self, tmp_path, db_engine):
        # 100 hours × 2 types (D, DF) = 200 rows
        fpath = _make_region_csv(tmp_path, n_hours=100)
        result = ingest_file(fpath, engine=db_engine, verbose=False)
        assert result["rows_ingested"] == 200

    def test_upsert_idempotent(self, tmp_path, db_engine):
        fpath = _make_region_csv(tmp_path, n_hours=50)
        r1 = ingest_file(fpath, engine=db_engine, verbose=False)
        r2 = ingest_file(fpath, engine=db_engine, verbose=False)
        # Second ingest should produce same count but not duplicate rows
        counts = get_table_counts(db_engine)
        assert counts["region_data"] == r1["rows_ingested"]

    def test_missing_file_raises(self, tmp_path, db_engine):
        with pytest.raises(FileNotFoundError):
            ingest_file(tmp_path / "nonexistent.csv", engine=db_engine)


# ─────────────────────────────────────────────────────────────────────────────
# ingest_file — other endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestOtherEndpoints:
    def test_fuel_type_ingest(self, tmp_path, db_engine):
        fpath = _make_fuel_csv(tmp_path)
        result = ingest_file(fpath, engine=db_engine, verbose=False)
        assert result["status"] == "success"
        assert result["rows_ingested"] == 200  # 50 hours × 4 fuels

    def test_interchange_ingest(self, tmp_path, db_engine):
        fpath = _make_interchange_csv(tmp_path)
        result = ingest_file(fpath, engine=db_engine, verbose=False)
        assert result["status"] == "success"
        assert result["rows_ingested"] == 50

    def test_sub_ba_ingest(self, tmp_path, db_engine):
        fpath = _make_subba_csv(tmp_path)
        result = ingest_file(fpath, engine=db_engine, verbose=False)
        assert result["status"] == "success"
        assert result["rows_ingested"] == 50


# ─────────────────────────────────────────────────────────────────────────────
# load_region_series / load_wide
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadHelpers:
    def test_load_region_series_returns_series(self, tmp_path, db_engine):
        fpath = _make_region_csv(tmp_path, ba="TST", n_hours=100)
        ingest_file(fpath, engine=db_engine, verbose=False)
        s = load_region_series(db_engine, "TST", dtype="D")
        assert isinstance(s, pd.Series)
        assert len(s) > 0
        assert s.index.tz is not None  # UTC-aware

    def test_load_wide_has_correct_columns(self, tmp_path, db_engine):
        fpath = _make_region_csv(tmp_path, ba="TST2", n_hours=100)
        ingest_file(fpath, engine=db_engine, verbose=False)
        df = load_wide(db_engine, "TST2", types=["D", "DF"])
        assert "D" in df.columns
        assert "DF" in df.columns
        assert len(df) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration test — real MISO file (skipped if file doesn't exist)
# ─────────────────────────────────────────────────────────────────────────────

REAL_MISO_FILE = Path("data/raw/region-data/MISO_20190101_20260409_region-data.csv")

@pytest.mark.skipif(
    not REAL_MISO_FILE.exists(),
    reason="Real MISO data file not available"
)
class TestRealMISOData:
    def test_real_miso_ingest_completes(self, tmp_path, db_engine):
        result = ingest_file(REAL_MISO_FILE, engine=db_engine, verbose=True)
        assert result["status"] == "success"
        assert result["rows_raw"] > 200_000
        assert result["rows_ingested"] > 0

    def test_real_miso_has_all_types(self, tmp_path, db_engine):
        ingest_file(REAL_MISO_FILE, engine=db_engine, verbose=False)
        s_d  = load_region_series(db_engine, "MISO", "D")
        s_df = load_region_series(db_engine, "MISO", "DF")
        assert len(s_d) > 50_000
        assert len(s_df) > 50_000
        # D and DF lengths very close (DF published 1h ahead so may differ by 1)
        assert abs(len(s_d) - len(s_df)) <= 2

    def test_real_miso_minimum_hours(self, tmp_path, db_engine):
        ingest_file(REAL_MISO_FILE, engine=db_engine, verbose=False)
        s = load_region_series(db_engine, "MISO", "D")
        assert len(s) >= 8760  # at least 1 year
