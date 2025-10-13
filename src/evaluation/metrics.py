"""
metrics.py — Evaluation metrics and walk-forward evaluator.

Metrics:
    mae(y_true, y_pred)    → float
    rmse(y_true, y_pred)   → float
    smape(y_true, y_pred)  → float  (percentage, 0-200)

Walk-forward evaluation:
    evaluate_folds(model, df, target_col, feature_cols)
        → DataFrame with per-fold + overall metrics for one model

    compare_models(models, df, target_col, feature_cols)
        → wide DataFrame comparing all models side by side

The fold column (from engineer.py) drives the expanding-window splits:
    fold == -1  →  2019 (anchor training data, always in train)
    fold == k   →  validated in fold k (trained on all folds < k)
    fold == 5   →  holdout — NEVER used here, excluded automatically
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Scalar metrics
# ─────────────────────────────────────────────────────────────────────────────


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def smape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Symmetric Mean Absolute Percentage Error (0-200 scale).

    sMAPE = 100 * mean(2 * |y - ŷ| / (|y| + |ŷ|))

    Rows where both y_true and y_pred are zero are excluded to avoid 0/0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (denom > 0)
    return float(100.0 * np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def score_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    n_rows: int | None = None,
) -> dict:
    """Compute MAE, RMSE, sMAPE for a prediction pair.

    Args:
        y_true: Actual demand values.
        y_pred: Predicted demand values.
        n_rows: Override row count (for reporting; defaults to len of valid pairs).

    Returns:
        dict with keys: mae, rmse, smape, n_rows
    """
    valid = y_true.notna() & y_pred.notna()
    yt = y_true[valid].to_numpy(dtype=float)
    yp = y_pred[valid].to_numpy(dtype=float)
    return {
        "mae": round(mae(yt, yp), 2),
        "rmse": round(rmse(yt, yp), 2),
        "smape": round(smape(yt, yp), 3),
        "n_rows": n_rows if n_rows is not None else int(valid.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward evaluation
# ─────────────────────────────────────────────────────────────────────────────

_TRAIN_FOLDS = [-1, 0, 1, 2, 3, 4]  # folds used for CV (holdout fold=5 excluded)
_VAL_FOLDS = [0, 1, 2, 3, 4]        # folds that appear as validation targets


def evaluate_folds(
    model,
    df: pd.DataFrame,
    target_col: str = "demand_mw",
    feature_cols: list[str] | None = None,
    fold_col: str = "fold",
) -> pd.DataFrame:
    """Walk-forward cross-validation for one model.

    For each validation fold k in [0..4]:
        train = df[df[fold_col] < k]   (expanding window, includes fold=-1)
        val   = df[df[fold_col] == k]
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score val predictions

    Holdout (fold=5) is never touched.

    Args:
        model:        A model with .fit(X, y) and .predict(X) methods.
        df:           Feature DataFrame (output of engineer.py).
        target_col:   Name of the target column.
        feature_cols: Columns to use as features (excludes target + meta cols).
                      If None, uses all columns except target, fold, respondent,
                      is_imputed, is_anomaly.
        fold_col:     Name of the fold column (default 'fold').

    Returns:
        DataFrame with columns [fold, mae, rmse, smape, n_rows] plus an
        'overall' row that aggregates all validation predictions.
    """
    if feature_cols is None:
        exclude = {target_col, fold_col, "respondent", "is_imputed", "is_anomaly"}
        feature_cols = [c for c in df.columns if c not in exclude]

    all_preds: list[pd.Series] = []
    all_actuals: list[pd.Series] = []
    rows = []

    for k in _VAL_FOLDS:
        train_mask = (df[fold_col] >= -1) & (df[fold_col] < k)
        val_mask = df[fold_col] == k

        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, target_col]
        X_val = df.loc[val_mask, feature_cols]
        y_val = df.loc[val_mask, target_col]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        scores = score_predictions(y_val, y_pred)
        scores["fold"] = k
        rows.append(scores)

        all_actuals.append(y_val)
        all_preds.append(y_pred)

    # Overall row (concat all val folds)
    if all_actuals:
        y_all = pd.concat(all_actuals)
        p_all = pd.concat(all_preds)
        overall = score_predictions(y_all, p_all)
        overall["fold"] = "overall"
        rows.append(overall)

    result = pd.DataFrame(rows, columns=["fold", "mae", "rmse", "smape", "n_rows"])
    result.insert(0, "model", getattr(model, "name", type(model).__name__))
    return result


def compare_models(
    models: list,
    df: pd.DataFrame,
    target_col: str = "demand_mw",
    feature_cols: list[str] | None = None,
    fold_col: str = "fold",
) -> pd.DataFrame:
    """Run walk-forward evaluation for multiple models and stack results.

    Args:
        models: List of model instances (each with fit/predict).
        df:     Feature DataFrame.
        target_col, feature_cols, fold_col: Passed to evaluate_folds.

    Returns:
        Stacked DataFrame with a 'model' column, one row per (model, fold).
        The 'overall' rows give the headline numbers for comparison.
    """
    frames = []
    for model in models:
        result = evaluate_folds(model, df, target_col, feature_cols, fold_col)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """Extract the 'overall' row for each model as a clean comparison table.

    Args:
        results: Output of compare_models().

    Returns:
        DataFrame sorted by RMSE ascending (best model first).
    """
    overall = results[results["fold"] == "overall"].copy()
    overall = overall.drop(columns=["fold"]).sort_values("rmse").reset_index(drop=True)
    overall.index += 1  # 1-based rank
    return overall
