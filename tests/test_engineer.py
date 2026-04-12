"""
tests/test_engineer.py — Test suite for src/features/engineer.py

Uses two synthetic in-memory SQLite BAs:
  TSTA — 3 years hourly data (2019-01-01 to 2021-12-31), has fuel data
  TSTB — 3 years hourly data (2019-01-01 to 2021-12-31), no fuel data
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from src.data.database import create_all_tables, get_engine
from src.features.engineer import (
    add_calendar_features,
    add_fourier_features,
    add_lag_features,
    add_rolling_features,
    assign_fold,
    build_features_all,
    build_features_for_ba,
    load_fuel_wide,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

_START = pd.Timestamp("2019-01-01", tz="UTC")
_END = pd.Timestamp("2021-12-31 23:00", tz="UTC")
_INDEX = pd.date_range(_START, _END, freq="h")
_N = len(_INDEX)  # ~26,304 hours

_FUEL_TYPES = ["WND", "SUN", "COL", "NG"]

EXPECTED_COLS = [
    "demand_mw", "eia_forecast_mw",
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_us_holiday",
    "sin_hour_1", "cos_hour_1", "sin_hour_2", "cos_hour_2",
    "sin_week_1", "cos_week_1", "sin_week_2", "cos_week_2",
    "sin_week_3", "cos_week_3",
    "lag_24h", "lag_48h", "lag_168h",
    "rolling_mean_24h", "rolling_std_24h",
    "rolling_mean_168h", "rolling_std_168h",
    "wind_pct", "solar_pct", "coal_pct", "ng_pct",
    "respondent", "is_imputed", "is_anomaly", "fold",
]


def _insert_region_data(conn, respondent: str, index: pd.DatetimeIndex) -> None:
    """Insert synthetic D and DF rows into region_data."""
    rng = np.random.default_rng(42)
    base_demand = 10_000.0 + rng.normal(0, 500, len(index))

    rows = []
    for i, ts in enumerate(index):
        period_str = ts.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        rows.append({
            "period": period_str,
            "respondent": respondent,
            "type": "D",
            "value_mwh": float(base_demand[i]),
            "is_anomaly": 0,
            "is_imputed": 0,
        })
        rows.append({
            "period": period_str,
            "respondent": respondent,
            "type": "DF",
            "value_mwh": float(base_demand[i] * 1.01),  # DF is ~1% above D
            "is_anomaly": 0,
            "is_imputed": 0,
        })

    conn.execute(
        text("""
            INSERT OR IGNORE INTO region_data
                (period, respondent, type, value_mwh, is_anomaly, is_imputed)
            VALUES
                (:period, :respondent, :type, :value_mwh, :is_anomaly, :is_imputed)
        """),
        rows,
    )


def _insert_fuel_data(conn, respondent: str, index: pd.DatetimeIndex) -> None:
    """Insert synthetic fuel_type_data for WND, SUN, COL, NG."""
    rng = np.random.default_rng(7)
    rows = []
    for ts in index:
        period_str = ts.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        for fuel, base in [("WND", 2000), ("SUN", 1000), ("COL", 3000), ("NG", 4000)]:
            rows.append({
                "period": period_str,
                "respondent": respondent,
                "fueltype": fuel,
                "value_mwh": float(abs(base + rng.normal(0, 100))),
            })

    conn.execute(
        text("""
            INSERT OR IGNORE INTO fuel_type_data
                (period, respondent, fueltype, value_mwh)
            VALUES
                (:period, :respondent, :fueltype, :value_mwh)
        """),
        rows,
    )


@pytest.fixture(scope="session")
def synth_engine():
    """SQLite engine with synthetic data for TSTA and TSTB.

    Uses a local _tmp/ path to avoid Windows system temp permission issues.
    """
    db_dir = Path(__file__).parent / "_tmp"
    db_dir.mkdir(exist_ok=True)
    db_path = db_dir / "test_features.db"
    db_path.unlink(missing_ok=True)  # clean leftover from prior run

    engine = get_engine(f"sqlite:///{db_path}")
    create_all_tables(engine)

    with engine.begin() as conn:
        _insert_region_data(conn, "TSTA", _INDEX)
        _insert_region_data(conn, "TSTB", _INDEX)
        _insert_fuel_data(conn, "TSTA", _INDEX)
        # TSTB intentionally has no fuel data

    yield engine

    engine.dispose()
    db_path.unlink(missing_ok=True)


@pytest.fixture()
def tsta_df(synth_engine, tmp_path):
    """Build feature matrix for TSTA (with fuel data)."""
    return build_features_for_ba(synth_engine, "TSTA", output_dir=tmp_path, save_parquet=False)


@pytest.fixture()
def tstb_df(synth_engine, tmp_path):
    """Build feature matrix for TSTB (no fuel data)."""
    return build_features_for_ba(synth_engine, "TSTB", output_dir=tmp_path, save_parquet=False)


# ─────────────────────────────────────────────────────────────────────────────
# TestFeatureMatrixShape
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureMatrixShape:

    def test_has_all_required_columns(self, tsta_df):
        for col in EXPECTED_COLS:
            assert col in tsta_df.columns, f"Missing column: {col}"

    def test_no_extra_unexpected_columns(self, tsta_df):
        assert set(tsta_df.columns) == set(EXPECTED_COLS), \
            f"Extra columns: {set(tsta_df.columns) - set(EXPECTED_COLS)}"

    def test_index_is_utc_datetimeindex(self, tsta_df):
        assert isinstance(tsta_df.index, pd.DatetimeIndex)
        assert tsta_df.index.tz is not None
        assert str(tsta_df.index.tz) == "UTC"

    def test_index_is_monotonic(self, tsta_df):
        assert tsta_df.index.is_monotonic_increasing

    def test_row_count_after_warmup_drop(self, tsta_df):
        # Should be _N - 168 warmup rows
        assert len(tsta_df) == _N - 168

    def test_dtypes_float32(self, tsta_df):
        float_cols = [c for c in EXPECTED_COLS if c not in
                      ("hour_of_day", "day_of_week", "month", "is_weekend",
                       "is_us_holiday", "is_imputed", "is_anomaly", "fold", "respondent")]
        for col in float_cols:
            assert tsta_df[col].dtype == np.float32, f"{col} dtype: {tsta_df[col].dtype}"

    def test_dtypes_int8(self, tsta_df):
        int8_cols = ["hour_of_day", "day_of_week", "month", "is_weekend",
                     "is_us_holiday", "is_imputed", "is_anomaly", "fold"]
        for col in int8_cols:
            assert tsta_df[col].dtype == np.int8, f"{col} dtype: {tsta_df[col].dtype}"

    def test_respondent_is_category(self, tsta_df):
        assert tsta_df["respondent"].dtype.name == "category"

    def test_empty_ba_returns_empty_frame(self, synth_engine, tmp_path):
        df = build_features_for_ba(synth_engine, "NONEXISTENT", output_dir=tmp_path, save_parquet=False)
        assert df.empty


# ─────────────────────────────────────────────────────────────────────────────
# TestNoLeakage
# ─────────────────────────────────────────────────────────────────────────────

class TestNoLeakage:

    def test_no_lag_1h_or_2h(self, tsta_df):
        # lag_1h and lag_2h have been removed — they leak future data for 24h-ahead
        assert "lag_1h" not in tsta_df.columns
        assert "lag_2h" not in tsta_df.columns

    def test_lag_24h_equals_t_minus_24(self, tsta_df):
        for pos in (50, 200, 500):
            assert tsta_df["lag_24h"].iloc[pos] == pytest.approx(
                tsta_df["demand_mw"].iloc[pos - 24]
            )

    def test_lag_168h_equals_t_minus_168(self, tsta_df):
        for pos in (200, 1000, 5000):
            assert tsta_df["lag_168h"].iloc[pos] == pytest.approx(
                tsta_df["demand_mw"].iloc[pos - 168]
            )

    def test_rolling_mean_24h_anchored_at_t_minus_24(self, tsta_df):
        # rolling_mean_24h at row T = mean(demand[T-47 : T-23])
        # Window covers 24h ending at T-24 (inclusive). All values ≥ 24h old.
        pos = 300
        expected = tsta_df["demand_mw"].iloc[pos - 47 : pos - 23].mean()
        actual = tsta_df["rolling_mean_24h"].iloc[pos]
        assert actual == pytest.approx(expected, rel=1e-4)

    def test_rolling_mean_168h_anchored_at_t_minus_24(self, tsta_df):
        # rolling_mean_168h at row T = mean(demand[T-191 : T-23])
        pos = 500
        expected = tsta_df["demand_mw"].iloc[pos - 191 : pos - 23].mean()
        actual = tsta_df["rolling_mean_168h"].iloc[pos]
        assert actual == pytest.approx(expected, rel=1e-4)

    def test_rolling_excludes_last_23h(self, tsta_df):
        # The rolling window ends at T-24, so demand at T through T-23 is NOT used.
        # Verify by checking the math: rolling_mean_24h at T equals mean of 24 values
        # from T-47 to T-24, not including anything more recent than T-24.
        pos = 400
        window = tsta_df["demand_mw"].iloc[pos - 47 : pos - 23]
        assert tsta_df["rolling_mean_24h"].iloc[pos] == pytest.approx(window.mean(), rel=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# TestFoldAssignment
# ─────────────────────────────────────────────────────────────────────────────

class TestFoldAssignment:

    def test_2019_rows_fold_minus_one(self, tsta_df):
        mask_2019 = tsta_df.index.year == 2019
        assert (tsta_df.loc[mask_2019, "fold"] == -1).all()

    def test_2020_rows_fold_zero(self, tsta_df):
        mask_2020 = tsta_df.index.year == 2020
        assert (tsta_df.loc[mask_2020, "fold"] == 0).all()

    def test_2021_rows_fold_one(self, tsta_df):
        mask_2021 = tsta_df.index.year == 2021
        assert (tsta_df.loc[mask_2021, "fold"] == 1).all()

    def test_2025_rows_fold_five(self):
        idx = pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC")
        folds = assign_fold(idx)
        assert (folds == 5).all()

    def test_2026_rows_fold_five_capped(self):
        idx = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
        folds = assign_fold(idx)
        assert (folds == 5).all()

    def test_fold_dtype_is_int8(self, tsta_df):
        assert tsta_df["fold"].dtype == np.int8

    def test_no_fold_values_outside_range(self, tsta_df):
        assert tsta_df["fold"].min() >= -1
        assert tsta_df["fold"].max() <= 5

    def test_assign_fold_2024(self):
        idx = pd.date_range("2024-06-01", periods=10, freq="h", tz="UTC")
        folds = assign_fold(idx)
        assert (folds == 4).all()


# ─────────────────────────────────────────────────────────────────────────────
# TestFuelMixFeatures
# ─────────────────────────────────────────────────────────────────────────────

class TestFuelMixFeatures:

    def test_fuel_pct_nonzero_when_data_exists(self, tsta_df):
        # TSTA has fuel data — at least some hours should have nonzero wind
        assert tsta_df["wind_pct"].sum() > 0

    def test_fuel_pct_all_zero_when_no_data(self, tstb_df):
        # TSTB has no fuel data — all pct columns must be 0
        for col in ("wind_pct", "solar_pct", "coal_pct", "ng_pct"):
            assert (tstb_df[col] == 0.0).all(), f"{col} has nonzero values for TSTB"

    def test_fuel_pct_range(self, tsta_df):
        for col in ("wind_pct", "solar_pct", "coal_pct", "ng_pct"):
            assert tsta_df[col].min() >= 0.0, f"{col} has negative values"
            assert tsta_df[col].max() <= 1.0, f"{col} exceeds 1.0"

    def test_fuel_pct_sum_lte_one(self, tsta_df):
        total = (
            tsta_df["wind_pct"]
            + tsta_df["solar_pct"]
            + tsta_df["coal_pct"]
            + tsta_df["ng_pct"]
        )
        # Sum of 4-fuel shares should be 1.0 (all gen comes from these 4)
        assert (total - 1.0).abs().max() < 1e-4

    def test_load_fuel_wide_returns_correct_columns(self, synth_engine):
        fuel = load_fuel_wide(synth_engine, "TSTA")
        assert set(fuel.columns) == {"WND", "SUN", "COL", "NG"}

    def test_load_fuel_wide_empty_for_missing_ba(self, synth_engine):
        fuel = load_fuel_wide(synth_engine, "NONEXISTENT")
        assert fuel.empty


# ─────────────────────────────────────────────────────────────────────────────
# TestParquetOutput
# ─────────────────────────────────────────────────────────────────────────────

class TestParquetOutput:

    def test_parquet_is_readable(self, synth_engine, tmp_path):
        build_features_for_ba(synth_engine, "TSTA", output_dir=tmp_path, save_parquet=True)
        path = tmp_path / "TSTA_features.parquet"
        assert path.exists()
        df = pd.read_parquet(path)
        assert not df.empty

    def test_parquet_index_is_datetimeindex(self, synth_engine, tmp_path):
        build_features_for_ba(synth_engine, "TSTA", output_dir=tmp_path, save_parquet=True)
        df = pd.read_parquet(tmp_path / "TSTA_features.parquet")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None

    def test_parquet_schema_columns_preserved(self, synth_engine, tmp_path):
        build_features_for_ba(synth_engine, "TSTA", output_dir=tmp_path, save_parquet=True)
        df = pd.read_parquet(tmp_path / "TSTA_features.parquet")
        for col in EXPECTED_COLS:
            assert col in df.columns, f"Missing after round-trip: {col}"

    def test_no_parquet_written_for_empty_ba(self, synth_engine, tmp_path):
        build_features_for_ba(synth_engine, "NONEXISTENT", output_dir=tmp_path, save_parquet=True)
        assert not (tmp_path / "NONEXISTENT_features.parquet").exists()


# ─────────────────────────────────────────────────────────────────────────────
# TestRollingStats
# ─────────────────────────────────────────────────────────────────────────────

class TestRollingStats:

    def test_rolling_mean_24h_manual(self, tsta_df):
        # rolling_mean_24h at T uses demand[T-47 : T-23] (24 values ending at T-24)
        pos = 500
        expected = float(tsta_df["demand_mw"].iloc[pos - 47 : pos - 23].mean())
        actual = float(tsta_df["rolling_mean_24h"].iloc[pos])
        assert abs(actual - expected) < 0.1  # float32 rounding tolerance

    def test_rolling_std_24h_manual(self, tsta_df):
        pos = 500
        expected = float(tsta_df["demand_mw"].iloc[pos - 47 : pos - 23].std())
        actual = float(tsta_df["rolling_std_24h"].iloc[pos])
        assert abs(actual - expected) < 1.0  # float32 rounding tolerance

    def test_rolling_mean_168h_manual(self, tsta_df):
        # rolling_mean_168h at T uses demand[T-191 : T-23] (168 values ending at T-24)
        pos = 500
        expected = float(tsta_df["demand_mw"].iloc[pos - 191 : pos - 23].mean())
        actual = float(tsta_df["rolling_mean_168h"].iloc[pos])
        assert abs(actual - expected) < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# TestHolidayDetection
# ─────────────────────────────────────────────────────────────────────────────

class TestHolidayDetection:

    def _make_calendar_df(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        df = pd.DataFrame({"demand_mw": 1.0}, index=dates)
        return add_calendar_features(df)

    def test_new_years_day_2019_is_holiday(self):
        idx = pd.date_range("2019-01-01", periods=24, freq="h", tz="UTC")
        df = self._make_calendar_df(idx)
        assert (df["is_us_holiday"] == 1).all()

    def test_christmas_2022_is_holiday(self):
        idx = pd.date_range("2022-12-25", periods=24, freq="h", tz="UTC")
        df = self._make_calendar_df(idx)
        assert (df["is_us_holiday"] == 1).all()

    def test_random_tuesday_is_not_holiday(self):
        idx = pd.date_range("2022-03-15", periods=24, freq="h", tz="UTC")
        df = self._make_calendar_df(idx)
        assert (df["is_us_holiday"] == 0).all()

    def test_all_24_hours_of_holiday_flagged(self):
        # All 24 hours of July 4, 2021 should be flagged
        idx = pd.date_range("2021-07-04", periods=24, freq="h", tz="UTC")
        df = self._make_calendar_df(idx)
        assert df["is_us_holiday"].sum() == 24


# ─────────────────────────────────────────────────────────────────────────────
# TestCalendarAndFourierNoNaN
# ─────────────────────────────────────────────────────────────────────────────

class TestCalendarAndFourierNoNaN:

    def test_calendar_features_no_nan(self, tsta_df):
        cal_cols = ["hour_of_day", "day_of_week", "month", "is_weekend", "is_us_holiday"]
        for col in cal_cols:
            assert tsta_df[col].isna().sum() == 0, f"NaN in {col}"

    def test_fourier_features_no_nan(self, tsta_df):
        fourier_cols = [c for c in EXPECTED_COLS if c.startswith(("sin_", "cos_"))]
        for col in fourier_cols:
            assert tsta_df[col].isna().sum() == 0, f"NaN in {col}"

    def test_fourier_values_bounded(self, tsta_df):
        fourier_cols = [c for c in EXPECTED_COLS if c.startswith(("sin_", "cos_"))]
        for col in fourier_cols:
            assert tsta_df[col].min() >= -1.0 - 1e-6, f"{col} below -1"
            assert tsta_df[col].max() <= 1.0 + 1e-6, f"{col} above 1"

    def test_hour_of_day_range(self, tsta_df):
        assert tsta_df["hour_of_day"].min() == 0
        assert tsta_df["hour_of_day"].max() == 23

    def test_day_of_week_range(self, tsta_df):
        assert tsta_df["day_of_week"].min() == 0
        assert tsta_df["day_of_week"].max() == 6

    def test_month_range(self, tsta_df):
        assert tsta_df["month"].min() == 1
        assert tsta_df["month"].max() == 12


# ─────────────────────────────────────────────────────────────────────────────
# TestCombinedOutput
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinedOutput:

    def test_all_features_has_multiple_bas(self, synth_engine, tmp_path):
        combined = build_features_all(synth_engine, ["TSTA", "TSTB"], output_dir=tmp_path, combined=True)
        assert combined["respondent"].nunique() == 2

    def test_combined_row_count(self, synth_engine, tmp_path):
        combined = build_features_all(synth_engine, ["TSTA", "TSTB"], output_dir=tmp_path, combined=True)
        # Both BAs have same number of rows
        assert len(combined) == (_N - 168) * 2

    def test_combined_parquet_written(self, synth_engine, tmp_path):
        build_features_all(synth_engine, ["TSTA", "TSTB"], output_dir=tmp_path, combined=True)
        assert (tmp_path / "ALL_features.parquet").exists()

    def test_no_combined_skips_all_parquet(self, synth_engine, tmp_path):
        build_features_all(synth_engine, ["TSTA"], output_dir=tmp_path, combined=False)
        assert not (tmp_path / "ALL_features.parquet").exists()


# ─────────────────────────────────────────────────────────────────────────────
# TestEdgeCases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_ba_with_no_d_series_returns_empty(self, synth_engine, tmp_path):
        df = build_features_for_ba(synth_engine, "NONEXISTENT", output_dir=tmp_path, save_parquet=False)
        assert df.empty

    def test_very_short_ba(self, tmp_path):
        """BA with exactly 300 hours → 132 rows after warmup drop."""
        idx = pd.date_range("2020-01-01", periods=300, freq="h", tz="UTC")
        db_path = tmp_path / "short.db"
        engine = get_engine(f"sqlite:///{db_path}")
        create_all_tables(engine)

        with engine.begin() as conn:
            rows = [
                {
                    "period": ts.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                    "respondent": "SHRT",
                    "type": t,
                    "value_mwh": 5000.0,
                    "is_anomaly": 0,
                    "is_imputed": 0,
                }
                for ts in idx
                for t in ("D", "DF")
            ]
            conn.execute(
                text("""
                    INSERT OR IGNORE INTO region_data
                        (period, respondent, type, value_mwh, is_anomaly, is_imputed)
                    VALUES
                        (:period, :respondent, :type, :value_mwh, :is_anomaly, :is_imputed)
                """),
                rows,
            )

        df = build_features_for_ba(engine, "SHRT", output_dir=tmp_path, save_parquet=False)
        assert len(df) == 300 - 168

    def test_assign_fold_standalone(self):
        """assign_fold works correctly with a mixed-year index."""
        idx = pd.DatetimeIndex([
            pd.Timestamp("2019-06-01", tz="UTC"),
            pd.Timestamp("2020-06-01", tz="UTC"),
            pd.Timestamp("2023-06-01", tz="UTC"),
            pd.Timestamp("2025-06-01", tz="UTC"),
        ])
        folds = assign_fold(idx)
        assert list(folds) == [-1, 0, 3, 5]

    def test_add_lag_features_standalone(self):
        """add_lag_features: lag_24h at position 100 equals demand at 76."""
        idx = pd.date_range("2020-01-01", periods=200, freq="h", tz="UTC")
        df = pd.DataFrame({"demand_mw": np.arange(200, dtype="float32")}, index=idx)
        add_lag_features(df)
        assert df["lag_24h"].iloc[100] == pytest.approx(df["demand_mw"].iloc[76])

    def test_add_rolling_features_anchored_at_t_minus_24(self):
        """rolling_mean_24h at pos T uses demand[T-47:T-23] — ends at T-24."""
        idx = pd.date_range("2020-01-01", periods=200, freq="h", tz="UTC")
        demand = np.ones(200, dtype="float32")
        demand[100] = 999.0  # spike at position 100
        df = pd.DataFrame({"demand_mw": demand}, index=idx)
        add_rolling_features(df)
        # At T=100: window covers demand[53:77] (positions 53..76) — spike at 100 not included
        assert df["rolling_mean_24h"].iloc[100] == pytest.approx(1.0, rel=1e-4)
        # At T=124: window covers demand[77:101] — demand[100]=999 IS included
        assert df["rolling_mean_24h"].iloc[124] != pytest.approx(1.0, rel=1e-1)
