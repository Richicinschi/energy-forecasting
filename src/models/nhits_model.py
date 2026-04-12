#!/usr/bin/env python3
"""N-HiTS (Neural Hierarchical Interpolation) model wrapper.

24h-ahead demand forecaster with all engineered features consumed as
historical exogenous variables. Uses NeuralForecast's cross_validation
with step_size=1 for proper walk-forward 24h-ahead evaluation.

Key design choices (vs the prior broken version):
- All features in X are passed via `hist_exog_list` so NHITS sees the same
  inputs Ridge/LightGBM see (lags, weather, fourier, fuel mix, etc.).
  The prior version only stored y and ignored every feature.
- Median imputation on training X; same medians reused on val X. NHITS is
  a neural net — NaN propagates and silently destroys training.
- Timestamps forced to tz-naive UTC throughout. Parquet index is tz-aware
  UTC; NeuralForecast strips tz internally and the lookup `pred_map` would
  100%-miss if we left them mismatched, collapsing every prediction to the
  train mean (suspected root cause of the constant-prediction bug).
- Built-in `scaler_type='robust'` — matches Ridge's RobustScaler choice
  and handles the COVID fold-0 anomaly that wrecks StandardScaler.
- Fallback for any NaN preds is `lag_24h` (24h persistence), not the train
  mean — never silently degrade to constant prediction again.
"""

import numpy as np
import pandas as pd


class _ImportanceShim:
    """Carries `feature_importances_` so train_models.py's _extract_importance()
    picks up NHITS scores. NHITS has no native importance — we populate this
    with an |Pearson corr(feature, y)| proxy in fit()."""

    def __init__(self, importances: np.ndarray):
        self.feature_importances_ = np.asarray(importances, dtype=float)


class NHITSModel:
    """N-HiTS for 24h-ahead demand forecasting with full exogenous features."""

    name = "N-HiTS"

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        import joblib
        return joblib.load(path)

    def __init__(
        self,
        h: int = 24,
        input_size: int = 168,
        n_blocks: list = None,
        mlp_units: list = None,
        n_pool_kernel_size: list = None,
        n_freq_downsample: list = None,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        max_steps: int = 5000,
        early_stop_patience_steps: int = 10,
        val_check_steps: int = 50,
        scaler_type: str = "robust",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.h = h
        self.input_size = input_size
        self.n_blocks = n_blocks or [2, 2, 2]
        self.mlp_units = mlp_units or [[256, 256], [256, 256], [256, 256]]
        self.n_pool_kernel_size = n_pool_kernel_size or [4, 2, 1]
        self.n_freq_downsample = n_freq_downsample or [4, 2, 1]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.early_stop_patience_steps = early_stop_patience_steps
        self.val_check_steps = val_check_steps
        self.scaler_type = scaler_type
        self.random_state = random_state
        self.verbose = verbose

        self._train_df: pd.DataFrame | None = None
        self._feature_cols: list[str] = []
        self._feature_medians: pd.Series | None = None
        self._is_fitted = False
        # Shim with .feature_importances_ so train_models.py's
        # _extract_importance() picks up a per-feature score. Populated in fit().
        self._estimator = _ImportanceShim(np.array([]))

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_naive_utc(idx) -> pd.DatetimeIndex:
        """Force to tz-naive UTC. Load-bearing — see module docstring."""
        di = pd.DatetimeIndex(idx)
        if di.tz is not None:
            di = di.tz_convert("UTC").tz_localize(None)
        return di

    # ─────────────────────────────────────────────────────────────────────
    # Fit / predict
    # ─────────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NHITSModel":
        """Store training rows + per-feature medians for imputation."""
        import logging, warnings, os
        os.environ["PYTORCH_LIGHTNING_DISABLE_PROGRESS_BAR"] = "1"
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("lightning").setLevel(logging.ERROR)
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
        logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        self._feature_cols = list(X.columns)
        self._feature_medians = X.median(numeric_only=True).fillna(0.0)
        X_imp = X.fillna(self._feature_medians)

        ds = self._to_naive_utc(X.index)
        cols = {
            "unique_id": np.full(len(X), "BA", dtype=object),
            "ds": ds,
            "y": np.asarray(y.values, dtype=float),
        }
        for c in self._feature_cols:
            cols[c] = np.asarray(X_imp[c].values, dtype=float)
        self._train_df = pd.DataFrame(cols)
        self._is_fitted = True

        # Importance proxy: |Pearson corr(feature, y)| on training data.
        # Not true NHITS importance (would need gradient/permutation), but a
        # cheap, stable signal that at least ranks features sensibly.
        y_vals = np.asarray(y.values, dtype=float)
        imps = np.zeros(len(self._feature_cols), dtype=float)
        for i, c in enumerate(self._feature_cols):
            x = np.asarray(X_imp[c].values, dtype=float)
            if np.std(x) > 0 and np.std(y_vals) > 0:
                imps[i] = abs(np.corrcoef(x, y_vals)[0, 1])
        imps = np.nan_to_num(imps, nan=0.0)
        self._estimator = _ImportanceShim(imps)

        if not self.verbose:
            print(f"  [{self.name}] fit: {len(X):,} rows, "
                  f"{len(self._feature_cols)} exog features")
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """24h-ahead prediction at every val timestamp via cross_validation."""
        import logging, warnings, os
        os.environ["PYTORCH_LIGHTNING_DISABLE_PROGRESS_BAR"] = "1"
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("lightning").setLevel(logging.ERROR)
        logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
        logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        if not self._is_fitted:
            raise RuntimeError(f"{self.name}: call fit() before predict()")

        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import NHITS
            from pytorch_lightning.callbacks import Callback
        except ImportError as e:
            raise ImportError(
                "neuralforecast is required. pip install neuralforecast"
            ) from e

        model_name = self.name
        progress_state = {"max_step": 0, "max_epoch": 0, "val_loss": float("nan")}

        class _StepProgress(Callback):
            def __init__(self, every_n: int = 250):
                self.every_n = every_n
                self._last_printed = -1

            def on_validation_epoch_end(self, trainer, pl_module):
                for key in ("ptl/val_loss", "val_loss", "valid_loss"):
                    v = trainer.callback_metrics.get(key)
                    if v is not None:
                        try:
                            progress_state["val_loss"] = float(v)
                        except Exception:
                            pass
                        break

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                step = trainer.global_step
                epoch = trainer.current_epoch
                progress_state["max_step"] = max(progress_state["max_step"], step)
                progress_state["max_epoch"] = max(progress_state["max_epoch"], epoch)
                if step > 0 and step % self.every_n == 0 and step != self._last_printed:
                    self._last_printed = step
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                    try:
                        loss_val = float(loss)
                    except Exception:
                        loss_val = float("nan")
                    vl = progress_state["val_loss"]
                    vl_str = f"{vl:.4f}" if vl == vl else "n/a"  # n/a if NaN
                    print(f"  [{model_name}] epoch {epoch} | step "
                          f"{step}/{trainer.max_steps}  "
                          f"train_loss={loss_val:.4f}  val_loss={vl_str}")

        missing_cols = [c for c in self._feature_cols if c not in X.columns]
        if missing_cols:
            raise ValueError(
                f"{self.name}: val X is missing fit-time columns: "
                f"{missing_cols[:5]}..."
            )
        X_imp = X[self._feature_cols].fillna(self._feature_medians)

        ds_val = self._to_naive_utc(X.index)

        # Reconstruct real val demand from lag_24h: lag_24h[t] = demand[t-24],
        # so looking up lag_24h at timestamp (t + 24h) gives demand[t].
        # This is NOT leakage — at walk-forward cutoff t we forecast t+24,
        # and demand[t] is by definition known to the operator at that moment.
        # val y is used only as AR input context (refit=False means no retraining),
        # and getting it right is load-bearing: a y=0 placeholder corrupts
        # TemporalNorm's per-window scaling and wrecks predictions.
        lag24 = X["lag_24h"]
        target_ts = X.index + pd.Timedelta(hours=24)
        y_recon = pd.Series(target_ts).map(lag24).to_numpy(dtype=float)
        # End-of-val rows (last 24) have no t+24 lookup — fall back to lag_24h at t
        fb = lag24.to_numpy(dtype=float)
        y_recon = np.where(np.isnan(y_recon), fb, y_recon)
        # Final fallback for any remaining NaN: training y median
        y_median = float(self._train_df["y"].median())
        y_recon = np.where(np.isnan(y_recon), y_median, y_recon)

        val_cols = {
            "unique_id": np.full(len(X), "BA", dtype=object),
            "ds": ds_val,
            "y": y_recon,
        }
        for c in self._feature_cols:
            val_cols[c] = np.asarray(X_imp[c].values, dtype=float)
        val = pd.DataFrame(val_cols)

        combined = pd.concat([self._train_df, val], ignore_index=True)
        combined = combined.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        nhits_kwargs = dict(
            h=self.h,
            input_size=self.input_size,
            hist_exog_list=self._feature_cols,
            stack_types=["identity", "identity", "identity"],
            n_blocks=self.n_blocks,
            mlp_units=self.mlp_units,
            n_pool_kernel_size=self.n_pool_kernel_size,
            n_freq_downsample=self.n_freq_downsample,
            pooling_mode="MaxPool1d",
            interpolation_mode="linear",
            dropout_prob_theta=0.1,
            activation="ReLU",
            scaler_type=self.scaler_type,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            max_steps=self.max_steps,
            early_stop_patience_steps=self.early_stop_patience_steps,
            val_check_steps=self.val_check_steps,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=True,
            callbacks=[_StepProgress(every_n=250)],
        )
        nhits = NHITS(**nhits_kwargs)
        nf = NeuralForecast(models=[nhits], freq="h")

        n_windows = len(X)
        if not self.verbose:
            print(f"  [{self.name}] predict: {n_windows:,} windows, step=1, "
                  f"refit=False, max_steps={self.max_steps}")

        # val_size carves the last N training timestamps as a held-out
        # validation set for early stopping (monitors ptl/val_loss).
        # Without this, EarlyStopping crashes with "metric not available".
        val_size = max(self.h * 7, min(int(0.1 * len(self._train_df)), 2000))
        cv_df = nf.cross_validation(
            df=combined,
            n_windows=n_windows,
            step_size=1,
            refit=False,
            val_size=val_size,
        )

        final_step = progress_state["max_step"]
        final_epoch = progress_state["max_epoch"]
        if 0 < final_step < self.max_steps:
            print(f"  [{self.name}] Early stopping triggered at step "
                  f"{final_step}/{self.max_steps} (epoch {final_epoch})")
        elif final_step >= self.max_steps:
            print(f"  [{self.name}] Ran all {self.max_steps} steps "
                  f"(epoch {final_epoch}) — no early stopping")
        else:
            print(f"  [{self.name}] (no training steps recorded)")

        cv_df = cv_df.copy()
        cv_df["horizon"] = (
            (cv_df["ds"] - cv_df["cutoff"]).dt.total_seconds() / 3600
        ).round().astype(int)
        cv_h = cv_df[cv_df["horizon"] == self.h]

        cv_h_ds = self._to_naive_utc(cv_h["ds"])
        pred_map = pd.Series(cv_h["NHITS"].to_numpy(), index=cv_h_ds)
        if not pred_map.index.is_unique:
            pred_map = pred_map.groupby(pred_map.index).mean()

        preds = pd.Series(ds_val).map(pred_map).to_numpy(dtype=float)

        n_missing = int(np.isnan(preds).sum())
        if n_missing:
            print(f"  WARNING: {n_missing}/{len(preds)} NaN preds "
                  f"— falling back to lag_24h")
            if "lag_24h" in X.columns:
                fb = X["lag_24h"].fillna(
                    self._feature_medians.get("lag_24h", 0.0)
                ).to_numpy(dtype=float)
                preds = np.where(np.isnan(preds), fb, preds)
            else:
                y_mean = float(self._train_df["y"].mean())
                preds = np.where(np.isnan(preds), y_mean, preds)

        if not self.verbose:
            print(f"  [{self.name}] Pred stats: "
                  f"mean={np.nanmean(preds):.0f}, std={np.nanstd(preds):.0f}, "
                  f"range=[{np.nanmin(preds):.0f}, {np.nanmax(preds):.0f}]")

        # Side-by-side RMSE vs reconstructed truth (y_recon).
        # y_recon is real demand at each val timestamp (from lag_24h.shift(-24)).
        # lag_24h is the 24h-persistence baseline.
        # eia_forecast_mw is excluded from features but may still be a column on X.
        def _rmse(yhat: np.ndarray, ytrue: np.ndarray) -> float:
            m = ~(np.isnan(yhat) | np.isnan(ytrue))
            if not m.any():
                return float("nan")
            return float(np.sqrt(np.mean((yhat[m] - ytrue[m]) ** 2)))

        rmse_nhits = _rmse(preds, y_recon)
        rmse_lag24 = _rmse(lag24.to_numpy(dtype=float), y_recon)
        eia_arr = (
            X["eia_forecast_mw"].to_numpy(dtype=float)
            if "eia_forecast_mw" in X.columns else None
        )
        rmse_eia = _rmse(eia_arr, y_recon) if eia_arr is not None else None

        vl = progress_state["val_loss"]
        vl_str = f"{vl:.4f}" if vl == vl else "n/a"
        eia_str = f"{rmse_eia:>8.0f}" if rmse_eia is not None else "    n/a "
        print(f"  [{self.name}] Summary | val_loss={vl_str} | "
              f"RMSE — NHITS={rmse_nhits:>8.0f}  lag24={rmse_lag24:>8.0f}  "
              f"EIA={eia_str}")

        return pd.Series(preds, index=X.index, name=self.name, dtype="float32")
