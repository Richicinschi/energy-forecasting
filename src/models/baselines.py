"""
baselines.py — Rule-based baseline models for energy demand forecasting.

All baselines implement a common sklearn-style interface (fit / predict)
so they can be swapped with ML models in the evaluation pipeline.

Four baselines, grouped by forecast horizon:

  1-hour-ahead horizon:
    Persistence1hModel    — ŷ(t) = y(t-1)          [lag_1h]     upper bound on accuracy

  24-hour-ahead horizon (the fair comparison against EIA DF):
    Persistence24hModel   — ŷ(t) = y(t-24)         [lag_24h]    naive 24h-ahead
    SeasonalNaiveModel    — ŷ(t) = y(t-168)         [lag_168h]   same hour last week
    EIAForecastModel      — ŷ(t) = EIA's DF column  [eia_forecast_mw]

The EIA day-ahead forecast is the primary benchmark — it is a 24h-ahead forecast,
so the fair naive comparison is Persistence24h (lag_24h), NOT lag_1h.
EIA DF (RMSE ~2,600 on MISO) beats Persistence24h (RMSE ~4,300) by ~40%.
Every ML model we build must beat EIA DF at the same 24h-ahead horizon.

Usage:
    from src.models.baselines import PersistenceModel, SeasonalNaiveModel, EIAForecastModel

    model = SeasonalNaiveModel()
    model.fit(X_train, y_train)   # no-op for baselines, required for interface compat
    preds = model.predict(X_test) # returns pd.Series aligned to X_test.index
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BaselineModel:
    """Abstract base for rule-based baselines.

    Subclasses must implement _predict_series(X).
    """

    name: str = "BaselineModel"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineModel":
        """No-op — baselines have no parameters to learn.

        Kept for sklearn Pipeline compatibility and to match the ML model API.
        """
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return predictions as a Series aligned to X.index.

        Args:
            X: Feature DataFrame from engineer.py. Must contain whichever
               columns the specific baseline requires (lag_1h, lag_168h,
               or eia_forecast_mw).

        Returns:
            pd.Series of float32 predictions, same index as X.
        """
        preds = self._predict_series(X)
        preds.name = self.name
        return preds.astype("float32")

    def _predict_series(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class Persistence1hModel(BaselineModel):
    """1-hour-ahead persistence: ŷ(t) = y(t-1).

    Uses the last known demand value. This is the practical upper bound
    on forecast accuracy — extremely hard to beat at 1h horizon because
    demand barely changes hour-to-hour. NOT a fair comparison against
    EIA's day-ahead forecast (which predicts 24h out).
    """

    name = "Persistence 1h-ahead"

    def _predict_series(self, X: pd.DataFrame) -> pd.Series:
        if "lag_1h" not in X.columns:
            raise ValueError("Persistence1hModel requires 'lag_1h' column in X")
        return X["lag_1h"].copy()


class Persistence24hModel(BaselineModel):
    """24-hour-ahead persistence: ŷ(t) = y(t-24).

    The fair naive baseline for comparing against EIA's day-ahead forecast.
    Both this model and EIA DF predict demand using only information available
    24+ hours before the target hour. On MISO this gives RMSE ~4,300 vs
    EIA DF RMSE ~2,600 — showing EIA's model adds real value over naive.
    """

    name = "Persistence 24h-ahead"

    def _predict_series(self, X: pd.DataFrame) -> pd.Series:
        if "lag_24h" not in X.columns:
            raise ValueError("Persistence24hModel requires 'lag_24h' column in X")
        return X["lag_24h"].copy()


class SeasonalNaiveModel(BaselineModel):
    """Weekly seasonal naive: ŷ(t) = y(t-168h).

    Predicts the same hour from exactly one week ago. Also a 24h+ ahead
    forecast (uses data from 168h prior). Compared against EIA DF to show
    whether weekly patterns alone can match a proper forecast model.
    """

    name = "Seasonal Naive 168h-ahead"

    def _predict_series(self, X: pd.DataFrame) -> pd.Series:
        if "lag_168h" not in X.columns:
            raise ValueError("SeasonalNaiveModel requires 'lag_168h' column in X")
        return X["lag_168h"].copy()


class EIAForecastModel(BaselineModel):
    """EIA day-ahead demand forecast baseline.

    Wraps the eia_forecast_mw column (the DF type from EIA-930) as a
    sklearn-compatible predictor. This is the primary benchmark — it
    represents EIA's own published day-ahead forecast, produced by their
    internal model. Any ML model we build must beat this to be useful.

    For the small number of rows where eia_forecast_mw is NaN (most recent
    hours where EIA hasn't published yet), falls back to lag_168h.
    """

    name = "EIA Day-Ahead Forecast (DF)"

    def _predict_series(self, X: pd.DataFrame) -> pd.Series:
        if "eia_forecast_mw" not in X.columns:
            raise ValueError("EIAForecastModel requires 'eia_forecast_mw' column in X")

        preds = X["eia_forecast_mw"].copy()

        # Fall back to lag_168h for any NaN eia_forecast rows
        nan_mask = preds.isna()
        if nan_mask.any() and "lag_168h" in X.columns:
            preds[nan_mask] = X.loc[nan_mask, "lag_168h"]

        return preds
