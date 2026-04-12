"""
ml_models.py — ML models for energy demand forecasting.

All models implement the same interface as baselines.py (fit / predict)
so they can be dropped into evaluate_folds() / compare_models() unchanged.

Six models, all targeting 24h-ahead demand:
    RidgeModel          — LinearRegression with L2 regularization + scaling + imputation
    MLPRegressorModel   — Feedforward neural net with scaling + imputation (sklearn)
    HistGBModel         — HistGradientBoostingRegressor (native NaN, CPU)
    XGBoostModel        — XGBRegressor (native NaN, CUDA GPU via device='cuda')
    LightGBMModel       — LGBMRegressor (native NaN, CPU)
    CatBoostModel       — CatBoostRegressor (native NaN, CUDA GPU via task_type='GPU')

Feature contract:
    X must be the feature DataFrame from engineer.py.
    Target column (demand_mw), fold, respondent, is_imputed, is_anomaly are
    excluded by the caller (metrics.evaluate_folds), NOT here.
    eia_forecast_mw is EXCLUDED from features — EIA's DF is submitted in the
    daily file (by 7 AM next day), not available 24h ahead at prediction time.
    It is used only as the EIA baseline model for comparison.

NaN handling:
    Ridge / MLP:  SimpleImputer(strategy='median') inside a Pipeline
    HistGB / XGB / LGBM / CatBoost: handle NaN natively (no imputer needed)

GPU acceleration (RTX 3060 Ti):
    XGBoost:  device='cuda'    (XGBoost 3.x syntax)
    CatBoost: task_type='GPU'  (requires CUDA toolkit)
    HistGB / LightGBM / Ridge / MLP: CPU only

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

# Deep learning models (optional dependencies)
try:
    from src.models.nhits_model import NHITSModel
except ImportError as e:
    print(f"Warning: NHITSModel not available - {e}")
    NHITSModel = None  # neuralforecast not installed


def _clip_5sigma(X):
    """Clip scaled features to [-5, 5] — prevents Ridge extrapolation explosions."""
    return np.clip(X, -5, 5)


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
    """Ridge regression with RobustScaler and median imputation.

    Pipeline: SimpleImputer(median) → RobustScaler → clip(±5) → Ridge(alpha=1.0)

    RobustScaler (median + IQR) replaces StandardScaler to handle distribution
    shifts across walk-forward folds — particularly fold 0 (COVID 2020) which
    has anomalous generation/interchange patterns that corrupt StandardScaler's
    mean/std estimates, causing explosive predictions on subsequent folds.

    The ±5 clip after scaling prevents any remaining extreme outliers from
    causing unbounded extrapolation.

    Median imputation is used instead of mean for the same outlier-robustness
    reason.
    """

    name = "Ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._estimator = None

    def _build_estimator(self):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("clip", FunctionTransformer(_clip_5sigma)),
            ("ridge", Ridge(alpha=self.alpha)),
        ])


class MLPRegressorModel(MLModel):
    """Feedforward neural network regressor (sklearn MLPRegressor).

    The only non-tree model in the lineup — provides a genuinely different
    inductive bias to complement gradient boosting models.

    Pipeline: SimpleImputer(median) → StandardScaler → MLPRegressor
    StandardScaler is mandatory: MLP training is extremely sensitive to
    feature scale — unscaled MW lags vs [-1,1] Fourier features would cause
    the optimizer to ignore small-scale features entirely.

    Architecture: 3 hidden layers (256 → 128 → 64) with ReLU activation.
    Early stopping on a 10% validation split prevents overfitting.

    CPU only (sklearn does not support GPU).
    """

    name = "MLP"

    def __init__(
        self,
        hidden_layer_sizes: tuple = (128, 64),
        max_iter: int = 200,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self._estimator = None

    def _build_estimator(self):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                max_iter=self.max_iter,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=self.random_state,
            )),
        ])


class HistGBModel(MLModel):
    """HistGradientBoosting regressor (sklearn, native NaN support).

    Fastest sklearn tree model — uses histogram-based splits with native
    NaN handling (no imputation needed). Comparable to LightGBM in speed.

    max_iter=300 with early stopping via validation_fraction=0.1.
    l2_regularization=0.1 for mild smoothing.
    CPU only (sklearn).
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
    """XGBoost regressor with CUDA GPU acceleration.

    XGBoost handles NaN by learning the optimal branch direction for missing
    values during split finding. No imputation needed.

    device='cuda' offloads histogram computation and tree building to GPU
    (RTX 3060 Ti). Requires CUDA toolkit and xgboost>=2.0.

    Requires xgboost>=2.0 (installed: 3.2.0).
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
            device="cuda",
            random_state=self.random_state,
            verbosity=0,
        )


class LightGBMModel(MLModel):
    """LightGBM regressor (native NaN support, fastest tree model).

    LightGBM's leaf-wise growth strategy often produces better accuracy than
    level-wise (XGBoost default) on tabular data at the same n_estimators.
    Native NaN support — no imputation needed.

    num_leaves=63 (matches HistGB max_leaf_nodes) for comparable model capacity.
    
    GPU support: Set device='gpu' for CUDA acceleration. Note: GPU has per-batch
    overhead, so CPU may be faster for small datasets (<50k rows).
    
    verbose=-1 suppresses training output.

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
        device: str = "cpu",  # "cpu" or "gpu"
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.device = device
        self._estimator = None

    def _build_estimator(self):
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise ImportError("lightgbm is required for LightGBMModel. pip install lightgbm") from e

        # Build params
        params = dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbose=-1,
        )
        
        # Add device-specific params
        if self.device == "gpu":
            params["device"] = "gpu"
            params["gpu_platform_id"] = 0
            params["gpu_device_id"] = 0
            # n_jobs not used for GPU
        else:
            params["n_jobs"] = -1

        return LGBMRegressor(**params)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Override to bypass sklearn's feature name validation.

        LightGBM 4.x auto-generates 'Column_0', 'Column_1'... feature names
        even when fitted with numpy arrays, and stores them on the booster.
        sklearn's predict() then compares these against the incoming array's
        feature names and warns when they don't match. Calling the underlying
        booster directly skips this check entirely.
        """
        if self._estimator is None:
            raise RuntimeError(f"{self.name}: call fit() before predict()")
        X_arr = X.to_numpy(dtype=float)
        # Call booster directly to avoid sklearn's _validate_data feature name check
        preds = self._estimator.booster_.predict(X_arr)
        return pd.Series(preds, index=X.index, name=self.name, dtype="float32")


class CatBoostModel(MLModel):
    """CatBoost regressor with CUDA GPU acceleration.

    CatBoost uses ordered boosting (a variant that reduces overfitting by
    preventing target leakage during training) — fundamentally different
    from standard gradient boosting in XGBoost/LightGBM.

    Native NaN support via built-in imputation during split finding.
    task_type='GPU' offloads training to GPU (RTX 3060 Ti).
    Model files are compact (~2MB each vs ~700MB for RF).

    Requires catboost>=1.2 (installed: 1.2.10).
    """

    name = "CatBoost"

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        random_seed: int = 42,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_seed = random_seed
        self._estimator = None

    def _build_estimator(self):
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise ImportError("catboost is required for CatBoostModel. pip install catboost") from e

        return CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            random_seed=self.random_seed,
            task_type="GPU",
            devices="0",
            verbose=0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class LightGBMGPUModel(LightGBMModel):
    """LightGBM with GPU acceleration."""
    name = "LightGBM-GPU"
    
    def __init__(self, **kwargs):
        super().__init__(device="gpu", **kwargs)


ALL_MODELS: dict[str, type[MLModel]] = {
    "ridge":        RidgeModel,
    "mlp":          MLPRegressorModel,
    "hist_gb":      HistGBModel,
    "xgboost":      XGBoostModel,        # GPU: RTX 3060 Ti via device='cuda'
    "lightgbm":     LightGBMModel,       # CPU
    "lightgbm_gpu": LightGBMGPUModel,    # GPU: CUDA via device='gpu'
    "catboost":     CatBoostModel,       # GPU: RTX 3060 Ti via task_type='GPU'
}

# Register N-HiTS if available
if NHITSModel is not None:
    ALL_MODELS["nhits"] = NHITSModel

# TabNet registered conditionally — requires pytorch-tabnet
try:
    from src.models.tabnet_model import TabNetModel
    ALL_MODELS["tabnet"] = TabNetModel
except ImportError:
    pass

# FT-Transformer registered conditionally — requires torch
try:
    from src.models.ft_transformer import FTTransformerModel
    ALL_MODELS["ft_transformer"] = FTTransformerModel
except ImportError:
    pass
