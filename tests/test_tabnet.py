"""
tests/test_tabnet.py — Tests for TabNetModel.

Kept separate from test_ml_models.py because TabNet requires pytorch-tabnet
and a CUDA-enabled PyTorch build. Tests are skipped automatically if either
dependency is missing so the rest of the suite always passes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.tabnet_model import TabNetModel


# ─────────────────────────────────────────────────────────────────────────────
# Skip guard — skip entire module if pytorch-tabnet not available
# ─────────────────────────────────────────────────────────────────────────────

pytabnet = pytest.importorskip(
    "pytorch_tabnet",
    reason="pytorch-tabnet not installed — skipping TabNet tests",
)

# pytorch-tabnet 4.1.0 uses scipy.sparse.base which was deprecated in scipy 1.12
# and pytorch-tabnet callbacks.py always emits a "Best weights" info message.
# Both originate in the library's internals — cannot be fixed without a patch.
# Suppress them narrowly here rather than broadly in pytest.ini.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Please import `spmatrix` from the `scipy.sparse` namespace"
        ":DeprecationWarning:scipy"
    ),
    pytest.mark.filterwarnings(
        "ignore:Best weights from best epoch are automatically used"
        ":UserWarning:pytorch_tabnet"
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_feature_df(seed: int = 42) -> pd.DataFrame:
    """Synthetic feature DataFrame covering fold -1, 0, 1 (2019-2021)."""
    idx = pd.date_range("2019-01-08", "2021-12-31 23:00", freq="h", tz="UTC")
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
    df["lag_24h"] = df["demand_mw"].shift(24).astype("float32")
    df["lag_168h"] = df["demand_mw"].shift(168).astype("float32")
    df["eia_forecast_mw"] = (demand + rng.normal(0, 50, n_hours)).astype("float32")
    df["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24).astype("float32")
    df["respondent"] = "TEST"
    df["is_imputed"] = np.int8(0)
    df["is_anomaly"] = np.int8(0)
    raw_fold = np.array(idx.year, dtype="int16") - 2020
    df["fold"] = np.clip(raw_fold, -1, 5).astype("int8")
    return df.iloc[168:].copy()


def _feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"demand_mw", "fold", "respondent", "is_imputed", "is_anomaly"}
    return [c for c in df.columns if c not in exclude]


@pytest.fixture(scope="module")
def small_df():
    return _make_feature_df()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTabNetInterface:
    def test_predict_before_fit_raises(self, small_df):
        model = TabNetModel()
        fcols = _feature_cols(small_df)
        with pytest.raises(RuntimeError, match="fit()"):
            model.predict(small_df[fcols])

    def test_fit_returns_self(self, small_df):
        model = TabNetModel(max_epochs=5, patience=3)
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        returned = model.fit(train[fcols], train["demand_mw"])
        assert returned is model

    def test_predict_returns_series_float32(self, small_df):
        model = TabNetModel(max_epochs=5, patience=3)
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        val = small_df[small_df["fold"] == 0]
        model.fit(train[fcols], train["demand_mw"])
        preds = model.predict(val[fcols])
        assert isinstance(preds, pd.Series)
        assert preds.dtype == "float32"

    def test_predict_index_matches_input(self, small_df):
        model = TabNetModel(max_epochs=5, patience=3)
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        val = small_df[small_df["fold"] == 0]
        model.fit(train[fcols], train["demand_mw"])
        preds = model.predict(val[fcols])
        pd.testing.assert_index_equal(preds.index, val.index)

    def test_predict_name_is_tabnet(self, small_df):
        model = TabNetModel(max_epochs=5, patience=3)
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        val = small_df[small_df["fold"] == 0]
        model.fit(train[fcols], train["demand_mw"])
        preds = model.predict(val[fcols])
        assert preds.name == "TabNet"

    def test_predict_no_nan_on_clean_input(self, small_df):
        model = TabNetModel(max_epochs=5, patience=3)
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1].dropna()
        val = small_df[small_df["fold"] == 0].dropna()
        model.fit(train[fcols], train["demand_mw"])
        preds = model.predict(val[fcols])
        assert preds.notna().all()


class TestTabNetNaN:
    def test_handles_nan_in_train(self, small_df):
        """SimpleImputer inside TabNetModel handles NaN in training data."""
        model = TabNetModel(max_epochs=5, patience=3)
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1][fcols + ["demand_mw"]].copy()
        train.iloc[::10, 0] = np.nan
        model.fit(train[fcols], train["demand_mw"])
        val = small_df[small_df["fold"] == 0]
        preds = model.predict(val[fcols])
        assert preds.notna().all()


class TestTabNetSerialization:
    def test_save_before_fit_raises(self, tmp_path):
        model = TabNetModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.save(tmp_path / "unfitted.joblib")

    def test_save_load_roundtrip(self, small_df, tmp_path):
        model = TabNetModel(max_epochs=5, patience=3)
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        val = small_df[small_df["fold"] == 0]
        model.fit(train[fcols], train["demand_mw"])

        save_path = tmp_path / "tabnet_test.joblib"
        model.save(save_path)
        assert save_path.exists()

        loaded = TabNetModel.load(save_path)
        preds_orig = model.predict(val[fcols]).to_numpy()
        preds_load = loaded.predict(val[fcols]).to_numpy()
        np.testing.assert_array_almost_equal(preds_orig, preds_load, decimal=4)

    def test_save_creates_parent_dirs(self, small_df, tmp_path):
        model = TabNetModel(max_epochs=5, patience=3)
        fcols = _feature_cols(small_df)
        train = small_df[small_df["fold"] == -1]
        model.fit(train[fcols], train["demand_mw"])
        nested = tmp_path / "nested" / "subdir" / "tabnet.joblib"
        model.save(nested)
        assert nested.exists()


class TestTabNetDevice:
    def test_uses_cuda_if_available(self):
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        model = TabNetModel()
        assert model._get_device() == device

    def test_model_name(self):
        assert TabNetModel.name == "TabNet"

    def test_custom_params(self):
        model = TabNetModel(n_d=32, n_steps=3, max_epochs=50)
        assert model.n_d == 32
        assert model.n_steps == 3
        assert model.max_epochs == 50
