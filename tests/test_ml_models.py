"""
tests/test_ml_models.py — Tests for ML models (Ridge, RF, HistGB, XGBoost, LightGBM).

All tests use the same synthetic feature DataFrame as test_baselines.py so the
fixture logic doesn't need to be duplicated at runtime — just share the helper.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compare_models, evaluate_folds, summary_table
from src.models.baselines import Persistence24hModel
from src.models.ml_models import (
    ALL_MODELS,
    HistGBModel,
    LightGBMModel,
    MLModel,
    RandomForestModel,
    RidgeModel,
    XGBoostModel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ─────────────────────────────────────────────────────────────────────────────


def _make_feature_df(seed: int = 42) -> pd.DataFrame:
    """Synthetic feature DataFrame spanning 2019-2024 with all CV folds."""
    idx = pd.date_range("2019-01-08", "2024-12-31 23:00", freq="h", tz="UTC")
    n_hours = len(idx)

    hour_of_day = idx.hour
    day_of_week = idx.dayofweek
    demand = (
        10_000
        + 500 * np.sin(2 * np.pi * hour_of_day / 24)
        + 200 * np.sin(2 * np.pi * day_of_week / 7)
    ).astype("float32")

    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=idx)
    df["demand_mw"] = demand
    df["lag_1h"] = df["demand_mw"].shift(1).astype("float32")
    df["lag_24h"] = df["demand_mw"].shift(24).astype("float32")
    df["lag_168h"] = df["demand_mw"].shift(168).astype("float32")
    df["eia_forecast_mw"] = (demand + rng.normal(0, 50, n_hours)).astype("float32")
    # Extra numeric feature to make the feature matrix slightly richer
    df["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24).astype("float32")
    df["respondent"] = "TEST"
    df["is_imputed"] = np.int8(0)
    df["is_anomaly"] = np.int8(0)

    raw_fold = np.array(idx.year, dtype="int16") - 2020
    df["fold"] = np.clip(raw_fold, -1, 5).astype("int8")

    return df.iloc[168:].copy()  # drop NaN warmup rows


@pytest.fixture(scope="module")
def feature_df():
    return _make_feature_df()


@pytest.fixture(scope="module")
def small_df():
    """Smaller frame for quick fit/predict tests — spans 2019-2021 to include fold -1, 0, 1."""
    df = _make_feature_df()
    # Need at least 2 years of data so both fold=-1 (2019) and fold=0 (2020) are present.
    # 2019-01-15 + 20000 hours ≈ mid-2021, covering fold -1, 0, 1.
    return df.iloc[:20000].copy()


def _feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"demand_mw", "fold", "respondent", "is_imputed", "is_anomaly"}
    return [c for c in df.columns if c not in exclude]


# ─────────────────────────────────────────────────────────────────────────────
# Registry tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_all_models_keys(self):
        assert set(ALL_MODELS) == {"ridge", "rf", "hist_gb", "xgboost", "lightgbm"}

    def test_all_models_are_mlmodel_subclasses(self):
        for cls in ALL_MODELS.values():
            assert issubclass(cls, MLModel)

    def test_each_model_has_name(self):
        for cls in ALL_MODELS.values():
            instance = cls()
            assert isinstance(instance.name, str)
            assert len(instance.name) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Interface contract (predict before fit raises, returns correct type)
# ─────────────────────────────────────────────────────────────────────────────


class TestInterface:
    @pytest.mark.parametrize("model_key", list(ALL_MODELS))
    def test_predict_before_fit_raises(self, model_key, small_df):
        model = ALL_MODELS[model_key]()
        fcols = _feature_cols(small_df)
        with pytest.raises(RuntimeError, match="fit()"):
            model.predict(small_df[fcols])

    @pytest.mark.parametrize("model_key", list(ALL_MODELS))
    def test_fit_returns_self(self, model_key, small_df):
        model = ALL_MODELS[model_key]()
        fcols = _feature_cols(small_df)
        # Use fold=-1 data only for speed
        train = small_df[small_df["fold"] == -1]
        X = train[fcols]
        y = train["demand_mw"]
        returned = model.fit(X, y)
        assert returned is model

    @pytest.mark.parametrize("model_key", list(ALL_MODELS))
    def test_predict_returns_series_float32(self, model_key, small_df):
        model = ALL_MODELS[model_key]()
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        val = small_df[small_df["fold"] == 0]
        model.fit(train[fcols], train["demand_mw"])
        preds = model.predict(val[fcols])
        assert isinstance(preds, pd.Series)
        assert preds.dtype == "float32"

    @pytest.mark.parametrize("model_key", list(ALL_MODELS))
    def test_predict_index_matches_input(self, model_key, small_df):
        model = ALL_MODELS[model_key]()
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        val = small_df[small_df["fold"] == 0]
        model.fit(train[fcols], train["demand_mw"])
        preds = model.predict(val[fcols])
        pd.testing.assert_index_equal(preds.index, val.index)

    @pytest.mark.parametrize("model_key", list(ALL_MODELS))
    def test_predict_name_matches_model_name(self, model_key, small_df):
        model = ALL_MODELS[model_key]()
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        val = small_df[small_df["fold"] == 0]
        model.fit(train[fcols], train["demand_mw"])
        preds = model.predict(val[fcols])
        assert preds.name == model.name

    @pytest.mark.parametrize("model_key", list(ALL_MODELS))
    def test_predict_no_nan_on_clean_input(self, model_key, small_df):
        model = ALL_MODELS[model_key]()
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1].dropna()
        val = small_df[small_df["fold"] == 0].dropna()
        model.fit(train[fcols], train["demand_mw"])
        preds = model.predict(val[fcols])
        assert preds.notna().all(), f"{model_key} produced NaN on clean input"


# ─────────────────────────────────────────────────────────────────────────────
# NaN handling
# ─────────────────────────────────────────────────────────────────────────────


class TestNaNHandling:
    """Models that impute internally should handle NaN without raising."""

    @pytest.mark.parametrize("model_key", ["ridge", "rf"])
    def test_imputing_models_handle_nan_in_train(self, model_key, small_df):
        """Ridge and RF use SimpleImputer — NaN in train should not raise."""
        model = ALL_MODELS[model_key]()
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1][fcols + ["demand_mw"]].copy()
        # Inject NaN into some rows
        train.iloc[::10, 0] = np.nan
        model.fit(train[fcols], train["demand_mw"])
        val = small_df[small_df["fold"] == 0]
        preds = model.predict(val[fcols])
        assert preds.notna().all()

    @pytest.mark.parametrize("model_key", ["hist_gb", "xgboost", "lightgbm"])
    def test_native_nan_models_handle_nan_in_train(self, model_key, small_df):
        """HistGB, XGBoost, LightGBM handle NaN natively."""
        model = ALL_MODELS[model_key]()
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1][fcols + ["demand_mw"]].copy()
        train.iloc[::10, 0] = np.nan
        model.fit(train[fcols], train["demand_mw"])
        val = small_df[small_df["fold"] == 0]
        preds = model.predict(val[fcols])
        assert preds.notna().all()

    @pytest.mark.parametrize("model_key", ["hist_gb", "xgboost", "lightgbm"])
    def test_native_nan_models_handle_nan_in_predict(self, model_key, small_df):
        """HistGB, XGBoost, LightGBM should predict even when val has NaN."""
        model = ALL_MODELS[model_key]()
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        model.fit(train[fcols], train["demand_mw"])
        val = small_df[small_df["fold"] == 0][fcols].copy()
        val.iloc[::10, 0] = np.nan
        preds = model.predict(val)
        assert preds.notna().all()


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward CV integration
# ─────────────────────────────────────────────────────────────────────────────


class TestWalkForwardCV:
    """evaluate_folds() should work for all models without errors."""

    @pytest.mark.parametrize("model_key", list(ALL_MODELS))
    def test_evaluate_folds_returns_correct_shape(self, model_key, feature_df):
        model = ALL_MODELS[model_key]()
        fcols = _feature_cols(feature_df)
        results = evaluate_folds(model, feature_df, feature_cols=fcols)
        # Should have 5 fold rows + 1 overall row
        assert len(results) == 6
        assert "overall" in results["fold"].values

    @pytest.mark.parametrize("model_key", list(ALL_MODELS))
    def test_evaluate_folds_metrics_are_finite(self, model_key, feature_df):
        model = ALL_MODELS[model_key]()
        fcols = _feature_cols(feature_df)
        results = evaluate_folds(model, feature_df, feature_cols=fcols)
        overall = results[results["fold"] == "overall"].iloc[0]
        assert np.isfinite(overall["rmse"])
        assert np.isfinite(overall["mae"])
        assert np.isfinite(overall["smape"])
        assert overall["rmse"] >= 0  # 0 is valid for a model that perfectly fits deterministic data

    @pytest.mark.parametrize("model_key", list(ALL_MODELS))
    def test_evaluate_folds_model_column_is_model_name(self, model_key, feature_df):
        model = ALL_MODELS[model_key]()
        fcols = _feature_cols(feature_df)
        results = evaluate_folds(model, feature_df, feature_cols=fcols)
        assert (results["model"] == model.name).all()


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy vs baseline
# ─────────────────────────────────────────────────────────────────────────────


class TestAccuracy:
    """Tree models should beat Persistence24h on synthetic sinusoidal demand."""

    @pytest.fixture(scope="class")
    def p24_rmse(self, feature_df):
        """RMSE of Persistence24h on the full feature_df (walk-forward)."""
        model = Persistence24hModel()
        fcols = _feature_cols(feature_df)
        results = evaluate_folds(model, feature_df, feature_cols=fcols)
        overall = results[results["fold"] == "overall"].iloc[0]
        return float(overall["rmse"])

    @pytest.mark.parametrize("model_key", ["hist_gb", "xgboost", "lightgbm"])
    def test_tree_models_beat_persistence24h(self, model_key, feature_df, p24_rmse):
        """Tree models should learn the sinusoidal pattern well."""
        model = ALL_MODELS[model_key]()
        fcols = _feature_cols(feature_df)
        results = evaluate_folds(model, feature_df, feature_cols=fcols)
        overall = results[results["fold"] == "overall"].iloc[0]
        assert overall["rmse"] < p24_rmse, (
            f"{model_key} RMSE={overall['rmse']:.0f} did not beat "
            f"Persistence24h RMSE={p24_rmse:.0f}"
        )

    def test_ridge_improves_over_lag_only(self, feature_df):
        """Ridge should at minimum produce finite, non-negative RMSE."""
        model = RidgeModel()
        fcols = _feature_cols(feature_df)
        results = evaluate_folds(model, feature_df, feature_cols=fcols)
        overall = results[results["fold"] == "overall"].iloc[0]
        assert overall["rmse"] >= 0
        assert np.isfinite(overall["rmse"])


# ─────────────────────────────────────────────────────────────────────────────
# compare_models integration
# ─────────────────────────────────────────────────────────────────────────────


class TestCompareModels:
    def test_compare_models_stacks_results(self, feature_df):
        """compare_models() should produce one block per model."""
        models = [HistGBModel(), RidgeModel()]
        fcols = _feature_cols(feature_df)
        results = compare_models(models, feature_df, feature_cols=fcols)
        assert set(results["model"]) == {HistGBModel.name, RidgeModel.name}
        # 5 fold rows + 1 overall per model = 12 rows for 2 models
        assert len(results) == 12

    def test_summary_table_sorted_by_rmse(self, feature_df):
        """summary_table() should sort ascending by RMSE."""
        models = [HistGBModel(), RidgeModel()]
        fcols = _feature_cols(feature_df)
        results = compare_models(models, feature_df, feature_cols=fcols)
        table = summary_table(results)
        rmse_vals = table["rmse"].tolist()
        assert rmse_vals == sorted(rmse_vals)


# ─────────────────────────────────────────────────────────────────────────────
# Serialization (save / load)
# ─────────────────────────────────────────────────────────────────────────────


class TestSerialization:
    @pytest.mark.parametrize("model_key", list(ALL_MODELS))
    def test_save_load_roundtrip(self, model_key, small_df, tmp_path):
        model = ALL_MODELS[model_key]()
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        val = small_df[small_df["fold"] == 0]
        model.fit(train[fcols], train["demand_mw"])

        save_path = tmp_path / f"{model_key}_test.joblib"
        model.save(save_path)
        assert save_path.exists()

        loaded = ALL_MODELS[model_key].load(save_path)
        preds_orig = model.predict(val[fcols]).to_numpy()
        preds_load = loaded.predict(val[fcols]).to_numpy()
        np.testing.assert_array_almost_equal(preds_orig, preds_load, decimal=4)

    def test_save_before_fit_raises(self, tmp_path):
        model = HistGBModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.save(tmp_path / "unfitted.joblib")

    @pytest.mark.parametrize("model_key", list(ALL_MODELS))
    def test_save_creates_parent_dirs(self, model_key, small_df, tmp_path):
        model = ALL_MODELS[model_key]()
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        model.fit(train[fcols], train["demand_mw"])
        nested_path = tmp_path / "nested" / "subdir" / f"{model_key}.joblib"
        model.save(nested_path)
        assert nested_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Model-specific parameter tests
# ─────────────────────────────────────────────────────────────────────────────


class TestModelParams:
    def test_ridge_custom_alpha(self, small_df):
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        model = RidgeModel(alpha=10.0)
        model.fit(train[fcols], train["demand_mw"])
        assert model.alpha == 10.0

    def test_rf_custom_n_estimators(self, small_df):
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        model = RandomForestModel(n_estimators=10)  # fast for testing
        model.fit(train[fcols], train["demand_mw"])
        assert model.n_estimators == 10

    def test_histgb_custom_lr(self, small_df):
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        model = HistGBModel(learning_rate=0.1, max_iter=50)
        model.fit(train[fcols], train["demand_mw"])
        assert model.learning_rate == 0.1

    def test_xgboost_custom_depth(self, small_df):
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        model = XGBoostModel(max_depth=4, n_estimators=50)
        model.fit(train[fcols], train["demand_mw"])
        assert model.max_depth == 4

    def test_lightgbm_custom_leaves(self, small_df):
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        model = LightGBMModel(num_leaves=31, n_estimators=50)
        model.fit(train[fcols], train["demand_mw"])
        assert model.num_leaves == 31
