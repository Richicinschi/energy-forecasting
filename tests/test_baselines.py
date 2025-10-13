"""
tests/test_baselines.py — Tests for baseline models and evaluation metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compare_models,
    evaluate_folds,
    mae,
    rmse,
    smape,
    summary_table,
)
from src.models.baselines import (
    EIAForecastModel,
    Persistence1hModel,
    Persistence24hModel,
    SeasonalNaiveModel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_feature_df(seed: int = 42) -> pd.DataFrame:
    """Synthetic feature DataFrame mimicking engineer.py output.

    Spans 2019-01-08 through 2024-12-31 to cover all CV folds (fold -1..4).
    """
    idx = pd.date_range("2019-01-08", "2024-12-31 23:00", freq="h", tz="UTC")
    n_hours = len(idx)

    # Sinusoidal demand with daily and weekly seasonality
    hour_of_day = idx.hour
    day_of_week = idx.dayofweek
    demand = (
        10_000
        + 500 * np.sin(2 * np.pi * hour_of_day / 24)
        + 200 * np.sin(2 * np.pi * day_of_week / 7)
    ).astype("float32")

    df = pd.DataFrame(index=idx)
    df["demand_mw"] = demand
    df["lag_1h"] = df["demand_mw"].shift(1).astype("float32")
    df["lag_24h"] = df["demand_mw"].shift(24).astype("float32")
    df["lag_168h"] = df["demand_mw"].shift(168).astype("float32")
    # EIA forecast: demand + small noise (slightly better than persistence)
    rng = np.random.default_rng(seed)
    df["eia_forecast_mw"] = (demand + rng.normal(0, 50, n_hours)).astype("float32")
    df["respondent"] = "TEST"
    df["is_imputed"] = np.int8(0)
    df["is_anomaly"] = np.int8(0)

    # Assign folds: fold = year - 2020 clipped to [-1, 5]
    raw_fold = np.array(idx.year, dtype="int16") - 2020
    df["fold"] = np.clip(raw_fold, -1, 5).astype("int8")

    # Drop warmup rows (lag_168h NaN)
    return df.iloc[168:].copy()


@pytest.fixture(scope="module")
def feature_df():
    return _make_feature_df()


@pytest.fixture(scope="module")
def small_df():
    # Short frame for fast model tests — just needs lag cols and folds
    df = _make_feature_df()
    return df.iloc[:2000].copy()


# ─────────────────────────────────────────────────────────────────────────────
# TestMetricFunctions
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricFunctions:

    def test_mae_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_mae_known_value(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        # errors: 2, 2, 3 → mean = 7/3
        assert mae(y_true, y_pred) == pytest.approx(7 / 3, rel=1e-5)

    def test_rmse_perfect_prediction(self):
        y = np.array([5.0, 10.0, 15.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # errors squared: 9, 16 → mean = 12.5 → sqrt = 3.535...
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5), rel=1e-5)

    def test_rmse_always_gte_mae(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(0, 1, 100)
        y_pred = rng.normal(0, 1, 100)
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)

    def test_smape_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0])
        assert smape(y, y) == pytest.approx(0.0)

    def test_smape_symmetric(self):
        y_true = np.array([100.0])
        y_pred = np.array([200.0])
        assert smape(y_true, y_pred) == pytest.approx(smape(y_pred, y_true))

    def test_smape_range(self):
        rng = np.random.default_rng(1)
        y_true = np.abs(rng.normal(10000, 1000, 200))
        y_pred = np.abs(rng.normal(10000, 2000, 200))
        s = smape(y_true, y_pred)
        assert 0.0 <= s <= 200.0

    def test_metrics_ignore_nan(self):
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 999.0, 3.0])
        # NaN in y_true should be ignored; result should be 0.0
        assert mae(y_true, y_pred) == pytest.approx(0.0)

    def test_metrics_ignore_inf(self):
        y_true = np.array([1.0, np.inf, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert mae(y_true, y_pred) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# TestPersistence1hModel
# ─────────────────────────────────────────────────────────────────────────────


class TestPersistence1hModel:

    def test_predict_returns_lag_1h(self, feature_df):
        model = Persistence1hModel()
        model.fit(feature_df, feature_df["demand_mw"])
        preds = model.predict(feature_df)
        pd.testing.assert_series_equal(
            preds.dropna(), feature_df["lag_1h"].dropna().astype("float32"),
            check_names=False,
        )

    def test_predict_returns_series(self, feature_df):
        preds = Persistence1hModel().predict(feature_df)
        assert isinstance(preds, pd.Series)

    def test_predict_index_matches(self, feature_df):
        preds = Persistence1hModel().predict(feature_df)
        assert preds.index.equals(feature_df.index)

    def test_predict_dtype_float32(self, feature_df):
        preds = Persistence1hModel().predict(feature_df)
        assert preds.dtype == np.float32

    def test_fit_returns_self(self, feature_df):
        model = Persistence1hModel()
        result = model.fit(feature_df, feature_df["demand_mw"])
        assert result is model

    def test_missing_column_raises(self, feature_df):
        df = feature_df.drop(columns=["lag_1h"])
        with pytest.raises(ValueError, match="lag_1h"):
            Persistence1hModel().predict(df)

    def test_name_contains_1h(self):
        assert "1h" in Persistence1hModel.name


# ─────────────────────────────────────────────────────────────────────────────
# TestPersistence24hModel
# ─────────────────────────────────────────────────────────────────────────────


class TestPersistence24hModel:

    def test_predict_returns_lag_24h(self, feature_df):
        model = Persistence24hModel()
        preds = model.predict(feature_df)
        pd.testing.assert_series_equal(
            preds.dropna(), feature_df["lag_24h"].dropna().astype("float32"),
            check_names=False,
        )

    def test_predict_dtype_float32(self, feature_df):
        assert Persistence24hModel().predict(feature_df).dtype == np.float32

    def test_predict_index_matches(self, feature_df):
        preds = Persistence24hModel().predict(feature_df)
        assert preds.index.equals(feature_df.index)

    def test_missing_column_raises(self, feature_df):
        df = feature_df.drop(columns=["lag_24h"])
        with pytest.raises(ValueError, match="lag_24h"):
            Persistence24hModel().predict(df)

    def test_name_contains_24h(self):
        assert "24h" in Persistence24hModel.name

    def test_worse_than_1h_persistence(self, feature_df):
        # 24h-ahead should always be harder (higher RMSE) than 1h-ahead
        from src.evaluation.metrics import rmse as _rmse
        y = feature_df["demand_mw"].dropna()
        p1 = Persistence1hModel().predict(feature_df).reindex(y.index)
        p24 = Persistence24hModel().predict(feature_df).reindex(y.index)
        valid = y.notna() & p1.notna() & p24.notna()
        assert _rmse(y[valid], p24[valid]) > _rmse(y[valid], p1[valid])


# ─────────────────────────────────────────────────────────────────────────────
# TestSeasonalNaiveModel
# ─────────────────────────────────────────────────────────────────────────────


class TestSeasonalNaiveModel:

    def test_predict_returns_lag_168h(self, feature_df):
        model = SeasonalNaiveModel()
        preds = model.predict(feature_df)
        pd.testing.assert_series_equal(
            preds.dropna(), feature_df["lag_168h"].dropna().astype("float32"),
            check_names=False,
        )

    def test_predict_dtype_float32(self, feature_df):
        assert SeasonalNaiveModel().predict(feature_df).dtype == np.float32

    def test_predict_index_matches(self, feature_df):
        preds = SeasonalNaiveModel().predict(feature_df)
        assert preds.index.equals(feature_df.index)

    def test_missing_column_raises(self, feature_df):
        df = feature_df.drop(columns=["lag_168h"])
        with pytest.raises(ValueError, match="lag_168h"):
            SeasonalNaiveModel().predict(df)

    def test_name_attribute(self):
        assert "168" in SeasonalNaiveModel.name or "Seasonal" in SeasonalNaiveModel.name


# ─────────────────────────────────────────────────────────────────────────────
# TestEIAForecastModel
# ─────────────────────────────────────────────────────────────────────────────


class TestEIAForecastModel:

    def test_predict_returns_eia_column(self, feature_df):
        # Rows where eia_forecast_mw is not NaN
        model = EIAForecastModel()
        preds = model.predict(feature_df)
        non_nan = feature_df["eia_forecast_mw"].notna()
        pd.testing.assert_series_equal(
            preds[non_nan],
            feature_df.loc[non_nan, "eia_forecast_mw"].astype("float32"),
            check_names=False,
        )

    def test_nan_fallback_to_lag_168h(self, feature_df):
        df = feature_df.copy()
        # Set some EIA values to NaN
        nan_rows = df.index[:10]
        df.loc[nan_rows, "eia_forecast_mw"] = np.nan
        preds = EIAForecastModel().predict(df)
        # Those rows should fall back to lag_168h
        for idx in nan_rows:
            if not np.isnan(df.loc[idx, "lag_168h"]):
                assert preds[idx] == pytest.approx(df.loc[idx, "lag_168h"])

    def test_predict_dtype_float32(self, feature_df):
        assert EIAForecastModel().predict(feature_df).dtype == np.float32

    def test_predict_index_matches(self, feature_df):
        preds = EIAForecastModel().predict(feature_df)
        assert preds.index.equals(feature_df.index)

    def test_missing_column_raises(self, feature_df):
        df = feature_df.drop(columns=["eia_forecast_mw"])
        with pytest.raises(ValueError, match="eia_forecast_mw"):
            EIAForecastModel().predict(df)

    def test_name_mentions_eia(self):
        assert "EIA" in EIAForecastModel.name


# ─────────────────────────────────────────────────────────────────────────────
# TestEvaluateFolds
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluateFolds:

    def test_returns_dataframe(self, feature_df):
        result = evaluate_folds(SeasonalNaiveModel(), feature_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_model_column(self, feature_df):
        result = evaluate_folds(SeasonalNaiveModel(), feature_df)
        assert "model" in result.columns

    def test_has_overall_row(self, feature_df):
        result = evaluate_folds(SeasonalNaiveModel(), feature_df)
        assert "overall" in result["fold"].values

    def test_has_metric_columns(self, feature_df):
        result = evaluate_folds(SeasonalNaiveModel(), feature_df)
        for col in ("mae", "rmse", "smape", "n_rows"):
            assert col in result.columns

    def test_holdout_fold_excluded(self, feature_df):
        result = evaluate_folds(SeasonalNaiveModel(), feature_df)
        assert 5 not in result["fold"].values

    def test_rmse_positive(self, feature_df):
        result = evaluate_folds(Persistence1hModel(), feature_df)
        overall = result[result["fold"] == "overall"].iloc[0]
        assert overall["rmse"] > 0

    def test_persistence24h_worse_than_1h(self, feature_df):
        # 24h-ahead is always harder than 1h-ahead
        r_1h = evaluate_folds(Persistence1hModel(), feature_df)
        r_24h = evaluate_folds(Persistence24hModel(), feature_df)
        rmse_1h = r_1h[r_1h["fold"] == "overall"]["rmse"].iloc[0]
        rmse_24h = r_24h[r_24h["fold"] == "overall"]["rmse"].iloc[0]
        assert rmse_24h > rmse_1h

    def test_eia_lower_rmse_than_persistence24h(self, feature_df):
        # EIA day-ahead forecast should beat naive 24h persistence
        r_eia = evaluate_folds(EIAForecastModel(), feature_df)
        r_24h = evaluate_folds(Persistence24hModel(), feature_df)
        rmse_eia = r_eia[r_eia["fold"] == "overall"]["rmse"].iloc[0]
        rmse_24h = r_24h[r_24h["fold"] == "overall"]["rmse"].iloc[0]
        assert rmse_eia < rmse_24h

    def test_n_rows_positive(self, feature_df):
        result = evaluate_folds(EIAForecastModel(), feature_df)
        assert (result["n_rows"] > 0).all()

    def test_custom_feature_cols(self, feature_df):
        # Should work with explicit feature col list
        result = evaluate_folds(
            Persistence1hModel(),
            feature_df,
            feature_cols=["lag_1h", "lag_24h", "lag_168h", "eia_forecast_mw"],
        )
        assert "overall" in result["fold"].values


# ─────────────────────────────────────────────────────────────────────────────
# TestCompareModels
# ─────────────────────────────────────────────────────────────────────────────


class TestCompareModels:

    def test_returns_all_models(self, feature_df):
        models = [Persistence1hModel(), Persistence24hModel(), SeasonalNaiveModel(), EIAForecastModel()]
        result = compare_models(models, feature_df)
        model_names = result["model"].unique()
        assert len(model_names) == 4

    def test_summary_table_sorted_by_rmse(self, feature_df):
        models = [Persistence1hModel(), Persistence24hModel(), SeasonalNaiveModel(), EIAForecastModel()]
        result = compare_models(models, feature_df)
        summary = summary_table(result)
        rmse_vals = summary["rmse"].tolist()
        assert rmse_vals == sorted(rmse_vals)

    def test_summary_table_no_fold_column(self, feature_df):
        models = [Persistence24hModel(), EIAForecastModel()]
        result = compare_models(models, feature_df)
        summary = summary_table(result)
        assert "fold" not in summary.columns

    def test_summary_table_has_four_rows(self, feature_df):
        models = [Persistence1hModel(), Persistence24hModel(), SeasonalNaiveModel(), EIAForecastModel()]
        result = compare_models(models, feature_df)
        summary = summary_table(result)
        assert len(summary) == 4

    def test_eia_model_name_in_summary(self, feature_df):
        models = [EIAForecastModel()]
        result = compare_models(models, feature_df)
        summary = summary_table(result)
        assert "EIA" in summary["model"].iloc[0]
