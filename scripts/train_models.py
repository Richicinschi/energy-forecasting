#!/usr/bin/env python3
"""
train_models.py — Train ML models via walk-forward CV and serialize to disk.

Reads per-BA Parquet feature files, runs walk-forward CV for each model+fold,
prints per-fold RMSE vs EIA DF baseline, and saves fitted models to
models/saved/{BA}_{model_name}_fold{k}.joblib.

Usage:
    # Train all models on MISO
    python scripts/train_models.py --ba MISO --model all

    # Train only HistGB and LightGBM on MISO
    python scripts/train_models.py --ba MISO --model hist_gb lightgbm

    # Train on multiple BAs
    python scripts/train_models.py --ba MISO PJM ERCO --model hist_gb

    # Train on all BAs (reads all *_features.parquet files)
    python scripts/train_models.py --ba ALL --model hist_gb

Available model keys: ridge, rf, hist_gb, xgboost, lightgbm
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.evaluation.metrics import _VAL_FOLDS, score_predictions
from src.models.baselines import EIAForecastModel, Persistence24hModel
from src.models.ml_models import ALL_MODELS

# Columns to exclude from features (target + metadata)
_EXCLUDE_COLS = {"demand_mw", "fold", "respondent", "is_imputed", "is_anomaly"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train ML models on energy demand feature matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model keys: ridge, rf, hist_gb, xgboost, lightgbm  (or: all)

Examples:
  python scripts/train_models.py --ba MISO --model all
  python scripts/train_models.py --ba MISO PJM --model hist_gb lightgbm
  python scripts/train_models.py --ba ALL --model hist_gb
        """,
    )
    parser.add_argument(
        "--ba", nargs="+", default=["ALL"], metavar="CODE",
        help="BA codes or ALL (default: ALL)",
    )
    parser.add_argument(
        "--model", nargs="+", default=["all"], metavar="KEY",
        help="Model key(s) or 'all' (default: all)",
    )
    parser.add_argument(
        "--features-dir", default=None, metavar="PATH",
        help="Directory containing {BA}_features.parquet files",
    )
    parser.add_argument(
        "--models-dir", default=None, metavar="PATH",
        help="Directory to save fitted models (default: models/saved/)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip serializing fitted models (useful for quick benchmarking)",
    )
    parser.add_argument(
        "--no-per-fold", action="store_true",
        help="Only print overall RMSE, not per-fold breakdown",
    )
    return parser.parse_args(argv)


def _resolve_models(model_keys: list[str]) -> dict[str, type]:
    """Resolve model keys to model classes. 'all' expands to all models."""
    if model_keys == ["all"]:
        return dict(ALL_MODELS)
    resolved = {}
    for key in model_keys:
        key = key.lower()
        if key not in ALL_MODELS:
            print(f"  Unknown model key '{key}'. Available: {list(ALL_MODELS)}")
            sys.exit(1)
        resolved[key] = ALL_MODELS[key]
    return resolved


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names (all cols except target and metadata)."""
    return [c for c in df.columns if c not in _EXCLUDE_COLS]


def _run_eia_baseline(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    """Run EIA DF baseline walk-forward and return overall RMSE."""
    from src.evaluation.metrics import evaluate_folds
    model = EIAForecastModel()
    results = evaluate_folds(model, df, feature_cols=feature_cols)
    overall = results[results["fold"] == "overall"]
    if overall.empty:
        return {"rmse": float("nan")}
    return {"rmse": float(overall["rmse"].iloc[0])}


def train_model_for_ba(
    model_key: str,
    model_cls,
    df: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path | None,
    ba: str,
    no_save: bool,
    no_per_fold: bool,
) -> dict:
    """Train a single model via walk-forward CV on one BA.

    Returns dict with overall mae, rmse, smape, n_rows.
    """
    model_name = model_cls().name  # get name before fitting
    t0 = time.perf_counter()

    all_actuals = []
    all_preds = []
    fold_rmses = {}

    for k in _VAL_FOLDS:
        train_mask = (df["fold"] >= -1) & (df["fold"] < k)
        val_mask = df["fold"] == k

        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, "demand_mw"]
        X_val = df.loc[val_mask, feature_cols]
        y_val = df.loc[val_mask, "demand_mw"]

        model = model_cls()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        scores = score_predictions(y_val, y_pred)
        fold_rmses[k] = int(round(scores["rmse"]))

        all_actuals.append(y_val)
        all_preds.append(y_pred)

        # Save last fold model (fold 4 = most recent before holdout)
        if not no_save and models_dir is not None and k == max(_VAL_FOLDS):
            save_path = models_dir / f"{ba}_{model_key}_fold{k}.joblib"
            model.save(save_path)

    elapsed = time.perf_counter() - t0

    # Overall metrics across all folds
    if not all_actuals:
        return {"mae": float("nan"), "rmse": float("nan"), "smape": float("nan"), "n_rows": 0}

    y_all = pd.concat(all_actuals)
    p_all = pd.concat(all_preds)
    overall = score_predictions(y_all, p_all)

    if not no_per_fold:
        print(f"    {model_name:<30}  RMSE={overall['rmse']:>7,.0f}  [{elapsed:.1f}s]")
        print(f"      per-fold RMSE: {fold_rmses}")
    else:
        print(f"    {model_name:<30}  RMSE={overall['rmse']:>7,.0f}  [{elapsed:.1f}s]")

    return overall


def main(argv=None) -> int:
    args = parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    features_dir = (
        Path(args.features_dir) if args.features_dir
        else project_root / "data" / "processed" / "features"
    )
    models_dir = (
        Path(args.models_dir) if args.models_dir
        else project_root / "models" / "saved"
    ) if not args.no_save else None

    if models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model classes
    models_to_run = _resolve_models(args.model)

    # Resolve BA list
    if args.ba == ["ALL"]:
        parquet_files = sorted(features_dir.glob("*_features.parquet"))
        parquet_files = [p for p in parquet_files if p.stem != "ALL_features"]
        ba_codes = [p.stem.replace("_features", "") for p in parquet_files]
    else:
        ba_codes = args.ba

    if not ba_codes:
        print("No BA feature files found. Run build_features.py first.")
        return 1

    print(f"\nML Training — {len(ba_codes)} BA(s), {len(models_to_run)} model(s)")
    print("=" * 70)

    aggregate_rows = []

    for ba in ba_codes:
        path = features_dir / f"{ba}_features.parquet"
        if not path.exists():
            print(f"  {ba}: no feature file found — skipping")
            continue

        df = pd.read_parquet(path)
        feature_cols = _get_feature_cols(df)

        print(f"\n  {ba}  ({len(df):,} rows, {len(feature_cols)} features)")
        print(f"  Feature cols: {feature_cols[:5]}... [{len(feature_cols)} total]")

        # EIA DF baseline for comparison
        eia_rmse = _run_eia_baseline(df, feature_cols)["rmse"]
        p24_model = Persistence24hModel()
        from src.evaluation.metrics import evaluate_folds
        p24_results = evaluate_folds(p24_model, df, feature_cols=feature_cols)
        p24_overall = p24_results[p24_results["fold"] == "overall"]
        p24_rmse = float(p24_overall["rmse"].iloc[0]) if not p24_overall.empty else float("nan")

        print(f"  Baselines:  EIA DF RMSE={eia_rmse:>7,.0f}   Persistence24h RMSE={p24_rmse:>7,.0f}")
        print()

        for model_key, model_cls in models_to_run.items():
            overall = train_model_for_ba(
                model_key, model_cls, df, feature_cols,
                models_dir, ba, args.no_save, args.no_per_fold,
            )
            beat = "BEAT ✓" if overall["rmse"] < eia_rmse else "below"
            pct = 100 * (overall["rmse"] - eia_rmse) / eia_rmse
            print(f"      vs EIA DF: {pct:+.1f}%  {beat}")
            aggregate_rows.append({
                "ba": ba,
                "model": model_cls().name,
                "mae": overall["mae"],
                "rmse": overall["rmse"],
                "smape": overall["smape"],
                "eia_rmse": eia_rmse,
                "vs_eia_pct": round(pct, 2),
            })
            print()

    if not aggregate_rows:
        print("\nNo results produced.")
        return 1

    # Aggregate summary
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY (mean across BAs)")
    print("=" * 70)
    agg_df = pd.DataFrame(aggregate_rows)
    summary = (
        agg_df.groupby("model")[["rmse", "mae", "smape", "vs_eia_pct"]]
        .mean()
        .round(2)
        .sort_values("rmse")
    )
    print(summary.to_string())

    if models_dir is not None:
        saved = list(models_dir.glob("*.joblib"))
        print(f"\nModels saved: {len(saved)} files in {models_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
