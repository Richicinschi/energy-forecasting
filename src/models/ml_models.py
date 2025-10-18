"""
ml_models.py — ML models for energy demand forecasting.

All models implement the same interface as baselines.py (fit / predict)
so they can be dropped into evaluate_folds() / compare_models() unchanged.

Five models, all targeting 24h-ahead demand:
    RidgeModel          — LinearRegression with L2 regularization + scaling + imputation
    RandomForestModel   — RandomForestRegressor with mean imputation
    HistGBModel         — HistGradientBoostingRegressor (native NaN support)
    XGBoostModel        — XGBRegressor (native NaN support, tree_method='hist')
    LightGBMModel       — LGBMRegressor (native NaN support)

Feature contract:
    X must be the feature DataFrame from engineer.py.
    Target column (demand_mw), fold, respondent, is_imputed, is_anomaly are
    excluded by the caller (metrics.evaluate_folds), NOT here.
    eia_forecast_mw IS included as a feature — EIA publishes it 24h ahead,
    so it is legitimately available at prediction time.

NaN handling:
    Ridge / RF:   SimpleImputer(strategy='median') inside a Pipeline
    HistGB / XGB / LGBM: handle NaN natively (no imputer needed)

Serialization:
    All models serialize with joblib. Use save() / load() class methods.
    Saved path convention: models/saved/{BA}_{model.name}_fold{k}.joblib

Usage:
    from src.models.ml_models import HistGBModel, ALL_MODELS

    model = HistGBModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_val)   # pd.Series float32 aligned to X_val.index

    # Serialize
    model.save("models/saved/MISO_HistGB_fold3.joblib")
    loaded = HistGBModel.load("models/saved/MISO_HistGB_fold3.joblib")
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────


class MLModel:
    """Abstract base for ML models.

    Provides the same fit/predict contract as BaselineModel so models are
    interchangeable in evaluate_folds() / compare_models().
    """

    name: str = "MLModel"

    # Subclasses set this to the wrapped estimator after fit()
    _estimator = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLModel":
        """Train the model. Returns self for chaining."""
        self._estimator = self._build_estimator()
        # Convert to numpy so tree libraries (LightGBM, XGBoost) don't register
        # feature names and then warn when predict() is called with numpy arrays.
        X_arr = X.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)
        self._estimator.fit(X_arr, y_arr)
        self._feature_names = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return predictions as float32 Series aligned to X.index."""
        if self._estimator is None:
            raise RuntimeError(f"{self.name}: call fit() before predict()")
        X_arr = X.to_numpy(dtype=float)
        preds = self._estimator.predict(X_arr)
        return pd.Series(preds, index=X.index, name=self.name, dtype="float32")

    def _build_estimator(self):
        """Return a fresh (unfitted) sklearn estimator or pipeline."""
        raise NotImplementedError

    def save(self, path: str | Path) -> Path:
        """Serialize the fitted model to disk with joblib."""
        if self._estimator is None:
            raise RuntimeError(f"{self.name}: cannot save — model not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": self._estimator, "feature_names": self._feature_names}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MLModel":
        """Load a serialized model and return a fitted instance."""
        path = Path(path)
        data = joblib.load(path)
        instance = cls()
        instance._estimator = data["estimator"]
        instance._feature_names = data.get("feature_names", [])
        return instance


# ─────────────────────────────────────────────────────────────────────────────
# Concrete models
# ─────────────────────────────────────────────────────────────────────────────


class RidgeModel(MLModel):
    """Ridge regression with StandardScaler and median imputation.

    Pipeline: SimpleImputer(median) → StandardScaler → Ridge(alpha=1.0)

    Median imputation is used instead of mean to be robust to outlier lags.
    Scaling is essential for Ridge so regularization is applied uniformly
    across features with very different magnitudes (lags in MW, Fourier in [-1,1]).
    """

    name = "Ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._estimator = None

    def _build_estimator(self):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha)),
        ])


class RandomForestModel(MLModel):
    """Random Forest regressor with median imputation.

    RandomForest does not handle NaN natively, so SimpleImputer(median) is
    prepended. Uses n_jobs=-1 for parallelism and max_features='sqrt' (default
    for regression in sklearn ≥1.1) to reduce overfitting.

    n_estimators=200 gives stable estimates without excessive training time.
    """

    name = "RandomForest"

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._estimator = None

    def _build_estimator(self):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(
                n_estimators=self.n_estimators,
                n_jobs=-1,
                random_state=self.random_state,
            )),
        ])


class HistGBModel(MLModel):
    """HistGradientBoosting regressor (sklearn, native NaN support).

    Fastest sklearn tree model — uses histogram-based splits with native
    NaN handling (no imputation needed). Comparable to LightGBM in speed.

    max_iter=300 with early stopping via validation_fraction=0.1.
    l2_regularization=0.1 for mild smoothing.
    """

    name = "HistGB"

    def __init__(
        self,
        max_iter: int = 300,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 63,
        l2_regularization: float = 0.1,
        random_state: int = 42,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self._estimator = None

    def _build_estimator(self):
        return HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )


class XGBoostModel(MLModel):
    """XGBoost regressor (native NaN support via tree_method='hist').

    XGBoost handles NaN by learning the optimal branch direction for missing
    values during split finding. No imputation needed.

    tree_method='hist' uses histogram-based splits (same algo as HistGB/LGBM)
    for speed on large tabular datasets. n_jobs=-1 for multi-core training.

    Requires xgboost>=1.7 (installed: 3.2.0).
    """

    name = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self._estimator = None

    def _build_estimator(self):
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError("xgboost is required for XGBoostModel. pip install xgboost") from e

        return XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            tree_method="hist",
            n_jobs=-1,
            random_state=self.random_state,
            verbosity=0,
        )


class LightGBMModel(MLModel):
    """LightGBM regressor (native NaN support, fastest tree model).

    LightGBM's leaf-wise growth strategy often produces better accuracy than
    level-wise (XGBoost default) on tabular data at the same n_estimators.
    Native NaN support — no imputation needed.

    num_leaves=63 (matches HistGB max_leaf_nodes) for comparable model capacity.
    n_jobs=-1 and verbose=-1 to suppress training output.

    Requires lightgbm>=4.0 (installed: 4.6.0).
    """

    name = "LightGBM"

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self._estimator = None

    def _build_estimator(self):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise ImportError("lightgbm is required for LightGBMModel. pip install lightgbm") from e

        return LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=-1,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

ALL_MODELS: dict[str, type[MLModel]] = {
    "ridge": RidgeModel,
    "rf": RandomForestModel,
    "hist_gb": HistGBModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
}
