"""
tabnet_model.py — TabNet regressor for energy demand forecasting.

TabNet (Arik & Pfister, 2019 — Google Research) uses sequential attention to
select which features to use at each decision step. Unlike trees that split on
one feature at a time, TabNet learns a sparse soft mask over all features,
giving both strong accuracy and interpretability: you can inspect which features
each prediction relied on.

Key properties:
    - GPU-native via PyTorch (RTX 3060 Ti via CUDA)
    - Native NaN support via learned masking
    - Attention masks are interpretable: model.explain() → per-sample feature importance
    - Competitive with tree ensembles on tabular data (NeurIPS 2019 benchmark)

Architecture (defaults):
    n_d=n_a=64    — width of decision/attention embedding
    n_steps=5     — number of sequential attention steps
    gamma=1.5     — coefficient for feature reusage (higher = less reuse)
    n_shared=2    — shared layers across steps
    momentum=0.02 — batch norm momentum (lower = more stable on small BAs)

Interface:
    Same fit(X, y) / predict(X) / save() / load() as all other MLModel subclasses.
    Drop into evaluate_folds() / compare_models() unchanged.

Requires:
    pytorch-tabnet>=4.0  (pip install pytorch-tabnet)
    torch>=2.0 with CUDA (pip install torch --index-url https://download.pytorch.org/whl/cu124)

Note on GPU:
    PyTorch CUDA wheels require Python <=3.12. This project runs Python 3.14,
    so TabNet runs on CPU. GPU support will work if the project is run under
    Python 3.12 with a CUDA-enabled torch build.

Usage:
    from src.models.tabnet_model import TabNetModel

    model = TabNetModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_val)   # pd.Series float32

    model.save("models/saved/MISO_TabNet_fold4.joblib")
    loaded = TabNetModel.load("models/saved/MISO_TabNet_fold4.joblib")
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class TabNetModel:
    """TabNet regressor with GPU support and the standard MLModel interface.

    NaN handling: SimpleImputer(median) before passing to TabNet.
    TabNet's attention mechanism still benefits from imputed values since
    it can learn to down-weight imputed features via the attention mask.

    StandardScaler is applied before TabNet — the batch normalization inside
    TabNet handles internal scale, but starting from normalized inputs helps
    convergence, especially for small BAs.
    """

    name = "TabNet"

    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_shared: int = 2,
        momentum: float = 0.02,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 8192,
        virtual_batch_size: int = 512,
        learning_rate: float = 2e-2,
        seed: int = 42,
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_shared = n_shared
        self.momentum = momentum
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.learning_rate = learning_rate
        self.seed = seed

        self._regressor = None
        self._imputer = None
        self._scaler = None
        self._feature_names: list[str] = []

    def _check_imports(self):
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pytorch-tabnet is required for TabNetModel.\n"
                "  pip install pytorch-tabnet\n"
                "Also ensure PyTorch with CUDA is installed:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
            ) from e

    def _get_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TabNetModel":
        """Train TabNet with imputation + scaling preprocessing."""
        self._check_imports()
        from pytorch_tabnet.tab_model import TabNetRegressor

        self._feature_names = list(X.columns)

        # Preprocess: impute then scale
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        X_arr = self._imputer.fit_transform(X.to_numpy(dtype=float))
        X_arr = self._scaler.fit_transform(X_arr).astype(np.float32)
        y_arr = y.to_numpy(dtype=float).reshape(-1, 1).astype(np.float32)

        device = self._get_device()

        self._regressor = TabNetRegressor(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_shared=self.n_shared,
            momentum=self.momentum,
            optimizer_params={"lr": self.learning_rate},
            seed=self.seed,
            device_name=device,
            verbose=0,
        )

        # Split off 10% validation for early stopping (same pattern as HistGB/MLP).
        # callbacks.py and abstract_model.py "Best weights" messages are patched
        # to print() so no UserWarnings are emitted.
        n_val = max(1, int(len(X_arr) * 0.1))
        X_val, y_val = X_arr[-n_val:], y_arr[-n_val:]
        X_tr, y_tr = X_arr[:-n_val], y_arr[:-n_val]

        self._regressor.fit(
            X_train=X_tr,
            y_train=y_tr,
            eval_set=[(X_val, y_val)],
            eval_name=["val"],
            eval_metric=["rmse"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return predictions as float32 Series aligned to X.index."""
        if self._regressor is None:
            raise RuntimeError(f"{self.name}: call fit() before predict()")

        X_arr = self._imputer.transform(X.to_numpy(dtype=float))
        X_arr = self._scaler.transform(X_arr).astype(np.float32)
        preds = self._regressor.predict(X_arr).flatten()
        return pd.Series(preds, index=X.index, name=self.name, dtype="float32")

    def save(self, path: str | Path) -> Path:
        """Serialize the fitted model to disk with joblib."""
        if self._regressor is None:
            raise RuntimeError(f"{self.name}: cannot save — model not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "regressor": self._regressor,
            "imputer": self._imputer,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "params": {
                "n_d": self.n_d, "n_a": self.n_a, "n_steps": self.n_steps,
                "gamma": self.gamma, "n_shared": self.n_shared,
                "momentum": self.momentum, "max_epochs": self.max_epochs,
                "patience": self.patience, "batch_size": self.batch_size,
                "virtual_batch_size": self.virtual_batch_size,
                "learning_rate": self.learning_rate, "seed": self.seed,
            },
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TabNetModel":
        """Load a serialized TabNet model."""
        path = Path(path)
        data = joblib.load(path)
        params = data.get("params", {})
        instance = cls(**params)
        instance._regressor = data["regressor"]
        instance._imputer = data["imputer"]
        instance._scaler = data["scaler"]
        instance._feature_names = data.get("feature_names", [])
        return instance
