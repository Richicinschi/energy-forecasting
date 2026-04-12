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

    # Train in parallel (4 workers) — safe for CPU models (ridge, mlp, hist_gb, lightgbm)
    python scripts/train_models.py --ba ALL --model mlp --workers 4

    # GPU models: keep --workers 1 (XGBoost/CatBoost share the GPU)
    python scripts/train_models.py --ba ALL --model xgboost --workers 1

Available model keys: ridge, mlp, hist_gb, xgboost, lightgbm, lightgbm_gpu, catboost
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.evaluation.metrics import _VAL_FOLDS, score_predictions
from src.models.baselines import EIAForecastModel, Persistence24hModel
from src.models.ml_models import ALL_MODELS

# Columns to exclude from features (target + metadata + EIA forecast).
# eia_forecast_mw is excluded because EIA's DF is submitted in the daily file
# (by 7 AM next day) — NOT available 24h ahead at prediction time. It is kept
# in the DataFrame only for the EIA baseline model comparison.
_EXCLUDE_COLS = {"demand_mw", "fold", "respondent", "is_imputed", "is_anomaly", "eia_forecast_mw"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train ML models on energy demand feature matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model keys: ridge, mlp, hist_gb, xgboost, lightgbm, catboost  (or: all)

Examples:
  python scripts/train_models.py --ba MISO --model all
  python scripts/train_models.py --ba MISO PJM --model hist_gb lightgbm
  python scripts/train_models.py --ba ALL --model hist_gb
  python scripts/train_models.py --ba ALL --model mlp --workers 4
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
    parser.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help="Number of parallel BA workers (default: 1). Use >1 for CPU-only models. "
             "Keep at 1 for GPU models (xgboost, catboost).",
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
    last_fold_model = None

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

        # Keep fold-k reference solely for feature-importance extraction.
        # Deployment model is a fresh refit on all pre-holdout data (below).
        if k == max(_VAL_FOLDS):
            last_fold_model = model

    # Refit on ALL pre-holdout data (folds [-1..max(_VAL_FOLDS)], excluding
    # holdout fold=5) and save that as the deployable artifact. This is
    # separate from the CV evaluation model by design.
    if not no_save and models_dir is not None:
        final_mask = (df["fold"] >= -1) & (df["fold"] <= max(_VAL_FOLDS))
        X_final = df.loc[final_mask, feature_cols]
        y_final = df.loc[final_mask, "demand_mw"]
        final_model = model_cls()
        final_model.fit(X_final, y_final)
        save_path = models_dir / f"{ba}_{model_key}_final.joblib"
        final_model.save(save_path)
        print(f"      [final-refit] {model_name} fit on {len(X_final):,} rows "
              f"— saved to {save_path.name}")

    elapsed = time.perf_counter() - t0

    # Overall metrics across all folds
    if not all_actuals:
        return {"mae": float("nan"), "rmse": float("nan"), "smape": float("nan"), "n_rows": 0,
                "feature_importance": None, "feature_names": feature_cols}

    y_all = pd.concat(all_actuals)
    p_all = pd.concat(all_preds)
    overall = score_predictions(y_all, p_all)

    if not no_per_fold:
        print(f"    {model_name:<30}  RMSE={overall['rmse']:>7,.0f}  [{elapsed:.1f}s]")
        print(f"      per-fold RMSE: {fold_rmses}")
    else:
        print(f"    {model_name:<30}  RMSE={overall['rmse']:>7,.0f}  [{elapsed:.1f}s]")

    # Extract feature importance from fold-4 model
    imp = _extract_importance(last_fold_model, feature_cols) if last_fold_model is not None else None
    overall["feature_importance"] = imp
    overall["feature_names"] = feature_cols
    return overall


def _extract_importance(model, feature_cols):
    """Extract normalised feature importance from a fitted model. Returns None if not available."""
    import numpy as np
    est = model._estimator
    try:
        if hasattr(est, "feature_importances_"):
            # Tree models: LightGBM, XGBoost, CatBoost
            imp = est.feature_importances_.astype(float)
        elif hasattr(est, "named_steps"):
            # Pipeline (Ridge): abs(coef * scaler.scale_)
            ridge = est.named_steps.get("ridge")
            scaler = est.named_steps.get("scaler")
            if ridge is None or scaler is None:
                return None
            imp = np.abs(ridge.coef_ * scaler.scale_).astype(float)
        else:
            return None
        total = imp.sum()
        if total > 0:
            imp = imp / total
        return imp.tolist()
    except Exception:
        return None


def _ba_worker(args: dict) -> dict:
    """Top-level worker function for ProcessPoolExecutor (must be picklable).

    Trains all requested models for one BA and returns a result dict.
    Output lines are collected as a string and printed by the caller.
    """
    import io
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

    import pandas as pd
    from src.evaluation.metrics import _VAL_FOLDS, evaluate_folds, score_predictions
    from src.models.baselines import EIAForecastModel, Persistence24hModel
    from src.models.ml_models import ALL_MODELS

    ba = args["ba"]
    model_keys = args["model_keys"]
    features_dir = _Path(args["features_dir"])
    models_dir = _Path(args["models_dir"]) if args["models_dir"] else None
    no_save = args["no_save"]
    no_per_fold = args["no_per_fold"]

    buf = io.StringIO()

    path = features_dir / f"{ba}_features.parquet"
    if not path.exists():
        buf.write(f"  {ba}: no feature file found — skipping\n")
        return {"ba": ba, "output": buf.getvalue(), "rows": []}

    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
    # Baselines need eia_forecast_mw and lag columns — use wider column set
    _baseline_exclude = {"demand_mw", "fold", "respondent", "is_imputed", "is_anomaly"}
    baseline_cols = [c for c in df.columns if c not in _baseline_exclude]

    buf.write(f"\n  {ba}  ({len(df):,} rows, {len(feature_cols)} features)\n")
    buf.write(f"  Feature cols: {feature_cols[:5]}... [{len(feature_cols)} total]\n")

    # Baselines (use baseline_cols which includes eia_forecast_mw)
    eia_model = EIAForecastModel()
    eia_results = evaluate_folds(eia_model, df, feature_cols=baseline_cols)
    eia_overall = eia_results[eia_results["fold"] == "overall"]
    eia_rmse = float(eia_overall["rmse"].iloc[0]) if not eia_overall.empty else float("nan")

    p24_model = Persistence24hModel()
    p24_results = evaluate_folds(p24_model, df, feature_cols=baseline_cols)
    p24_overall = p24_results[p24_results["fold"] == "overall"]
    p24_rmse = float(p24_overall["rmse"].iloc[0]) if not p24_overall.empty else float("nan")

    buf.write(f"  Baselines:  EIA DF RMSE={eia_rmse:>7,.0f}   Persistence24h RMSE={p24_rmse:>7,.0f}\n\n")

    agg_rows = []
    for model_key in model_keys:
        model_cls = ALL_MODELS[model_key]
        model_name = model_cls().name
        t0 = time.perf_counter()

        all_actuals, all_preds, fold_rmses = [], [], {}
        last_fold_model = None
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

            if k == max(_VAL_FOLDS):
                last_fold_model = model

        # Refit on ALL pre-holdout data (folds [-1..max(_VAL_FOLDS)]).
        if not no_save and models_dir is not None:
            final_mask = (df["fold"] >= -1) & (df["fold"] <= max(_VAL_FOLDS))
            X_final = df.loc[final_mask, feature_cols]
            y_final = df.loc[final_mask, "demand_mw"]
            final_model = model_cls()
            final_model.fit(X_final, y_final)
            save_path = models_dir / f"{ba}_{model_key}_final.joblib"
            final_model.save(save_path)
            buf.write(f"      [final-refit] {model_name} fit on "
                      f"{len(X_final):,} rows — saved to {save_path.name}\n")

        elapsed = time.perf_counter() - t0
        if not all_actuals:
            continue

        y_all = pd.concat(all_actuals)
        p_all = pd.concat(all_preds)
        overall = score_predictions(y_all, p_all)

        buf.write(f"    {model_name:<30}  RMSE={overall['rmse']:>7,.0f}  [{elapsed:.1f}s]\n")
        if not no_per_fold:
            buf.write(f"      per-fold RMSE: {fold_rmses}\n")

        beat = "BEAT" if overall["rmse"] < eia_rmse else "below"
        pct = 100 * (overall["rmse"] - eia_rmse) / eia_rmse
        buf.write(f"      vs EIA DF: {pct:+.1f}%  {beat}\n\n")

        imp = _extract_importance(last_fold_model, feature_cols)
        agg_rows.append({
            "ba": ba,
            "model": model_name,
            "mae": overall["mae"],
            "rmse": overall["rmse"],
            "smape": overall["smape"],
            "eia_rmse": eia_rmse,
            "p24_rmse": p24_rmse,
            "vs_eia_pct": round(pct, 2),
            "feature_importance": imp,
            "feature_names": feature_cols,
        })

    return {"ba": ba, "output": buf.getvalue(), "rows": agg_rows}


def _format_elapsed(seconds: float) -> str:
    """Format elapsed time in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}m{s:.0f}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h}h{m}m{s:.0f}s"


def main(argv=None) -> int:
    script_start_time = time.perf_counter()
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

    models_to_run = _resolve_models(args.model)

    if args.ba == ["ALL"]:
        parquet_files = sorted(features_dir.glob("*_features.parquet"))
        parquet_files = [p for p in parquet_files if p.stem != "ALL_features"]
        ba_codes = [p.stem.replace("_features", "") for p in parquet_files]
    else:
        ba_codes = args.ba

    if not ba_codes:
        print("No BA feature files found. Run build_features.py first.")
        return 1

    workers = min(args.workers, len(ba_codes))
    print(f"\nML Training — {len(ba_codes)} BA(s), {len(models_to_run)} model(s), {workers} worker(s)")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    aggregate_rows = []

    if workers == 1:
        # Sequential path — same as before, prints live
        for ba in ba_codes:
            path = features_dir / f"{ba}_features.parquet"
            if not path.exists():
                print(f"  {ba}: no feature file found — skipping")
                continue

            df = pd.read_parquet(path)
            feature_cols = _get_feature_cols(df)
            # Baselines need eia_forecast_mw and lag columns — use wider column set
            _bl_exclude = {"demand_mw", "fold", "respondent", "is_imputed", "is_anomaly"}
            baseline_cols = [c for c in df.columns if c not in _bl_exclude]

            print(f"\n  {ba}  ({len(df):,} rows, {len(feature_cols)} features)")
            print(f"  Feature cols: {feature_cols[:5]}... [{len(feature_cols)} total]")

            eia_rmse = None
            eia_model = EIAForecastModel()
            from src.evaluation.metrics import evaluate_folds
            eia_results = evaluate_folds(eia_model, df, feature_cols=baseline_cols)
            eia_overall = eia_results[eia_results["fold"] == "overall"]
            eia_rmse = float(eia_overall["rmse"].iloc[0]) if not eia_overall.empty else float("nan")

            p24_model = Persistence24hModel()
            p24_results = evaluate_folds(p24_model, df, feature_cols=baseline_cols)
            p24_overall = p24_results[p24_results["fold"] == "overall"]
            p24_rmse = float(p24_overall["rmse"].iloc[0]) if not p24_overall.empty else float("nan")

            print(f"  Baselines:  EIA DF RMSE={eia_rmse:>7,.0f}   Persistence24h RMSE={p24_rmse:>7,.0f}")
            print()

            for model_key, model_cls in models_to_run.items():
                overall = train_model_for_ba(
                    model_key, model_cls, df, feature_cols,
                    models_dir, ba, args.no_save, args.no_per_fold,
                )
                beat = "BEAT" if overall["rmse"] < eia_rmse else "below"
                pct = 100 * (overall["rmse"] - eia_rmse) / eia_rmse
                print(f"      vs EIA DF: {pct:+.1f}%  {beat}")
                aggregate_rows.append({
                    "ba": ba,
                    "model": model_cls().name,
                    "mae": overall["mae"],
                    "rmse": overall["rmse"],
                    "smape": overall["smape"],
                    "eia_rmse": eia_rmse,
                    "p24_rmse": p24_rmse,
                    "vs_eia_pct": round(pct, 2),
                    "feature_importance": overall.get("feature_importance"),
                    "feature_names": overall.get("feature_names", feature_cols),
                })
                print()

    else:
        # Parallel path — N BAs in parallel, output printed as each completes
        worker_args = [
            {
                "ba": ba,
                "model_keys": list(models_to_run.keys()),
                "features_dir": str(features_dir),
                "models_dir": str(models_dir) if models_dir else None,
                "no_save": args.no_save,
                "no_per_fold": args.no_per_fold,
            }
            for ba in ba_codes
        ]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_ba_worker, wa): wa["ba"] for wa in worker_args}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                print(result["output"], end="", flush=True)
                print(f"  [{completed}/{len(ba_codes)} BAs done]", flush=True)
                aggregate_rows.extend(result["rows"])

    if not aggregate_rows:
        print("\nNo results produced.")
        return 1

    agg_df = pd.DataFrame(aggregate_rows)

    # ── Aggregate summary ────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - script_start_time
    print("\n" + "=" * 78)
    print("AGGREGATE SUMMARY")
    print(f"Total elapsed time: {_format_elapsed(elapsed_total)}")
    print("=" * 78)

    unique_bas = agg_df["ba"].unique()
    if len(unique_bas) == 1:
        ba = unique_bas[0]
        rows = agg_df[agg_df["ba"] == ba]
        eia_r = float(rows["eia_rmse"].iloc[0])
        p24_r = float(rows["p24_rmse"].iloc[0]) if "p24_rmse" in rows.columns else float("nan")

        def _pct(x: float, base: float) -> str:
            if not base or base != base:
                return "   n/a "
            p = 100 * (x - base) / base
            sign = "+" if p >= 0 else ""
            return f"{sign}{p:>5.1f}%"

        print(f"\n  SINGLE-BA HEAD-TO-HEAD — {ba}")
        print("  " + "─" * 60)
        print(f"    {'EIA DF baseline':<22}: {eia_r:>9,.0f} MW")
        if p24_r == p24_r:  # not NaN
            print(f"    {'Persistence24h':<22}: {p24_r:>9,.0f} MW   "
                  f"({_pct(p24_r, eia_r)} vs EIA)")
        for _, r in rows.sort_values("rmse").iterrows():
            beat = "BEAT" if r["rmse"] < eia_r else "below"
            print(f"    {r['model']:<22}: {r['rmse']:>9,.0f} MW   "
                  f"({_pct(r['rmse'], eia_r)} vs EIA)  [{beat}]")

    for model_name, grp in agg_df.groupby("model"):
        wins     = int((grp["rmse"] < grp["eia_rmse"]).sum())
        n        = len(grp)

        # Headline stats over ALL BAs — no exclusion. Explosions are called
        # out separately but are included in these numbers.
        mean_rmse   = float(grp["rmse"].mean())
        median_rmse = float(grp["rmse"].median())
        mean_eia    = float(grp["eia_rmse"].mean())
        median_eia  = float(grp["eia_rmse"].median())
        mean_vs     = 100 * (mean_rmse - mean_eia) / mean_eia if mean_eia else float("nan")
        median_vs   = float(grp["vs_eia_pct"].median())
        m_mae       = float(grp["mae"].mean())
        m_smape     = float(grp["smape"].mean())

        exploded = grp[grp["rmse"] >= grp["eia_rmse"] * 50]

        def _s(p: float) -> str:
            return f"+{p:.1f}%" if p >= 0 else f"{p:.1f}%"

        model_time = elapsed_total / len(agg_df["model"].unique())

        print(f"\n  {model_name}")
        print(f"    Win rate  : {wins}/{n}  ({100*wins/max(n,1):.0f}%)")
        print(f"    Mean      : Model={mean_rmse:>9,.0f}  EIA={mean_eia:>9,.0f}  ({_s(mean_vs)})")
        print(f"    Median    : Model={median_rmse:>9,.0f}  EIA={median_eia:>9,.0f}  ({_s(median_vs)})")
        if len(exploded):
            expl_list = ", ".join(
                f"{r.ba}({r.rmse/r.eia_rmse:.0f}x)" for _, r in exploded.iterrows()
            )
            print(f"    Explosions: {len(exploded)} BAs (RMSE > 50× EIA): {expl_list}")
        print(f"    MAE       : {m_mae:>9,.0f} MW")
        print(f"    SMAPE     : {m_smape:>9.2f}%")
        print(f"    Time      : ~{_format_elapsed(model_time)} (estimated)")

    # ── Per-BA table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PER-BA BREAKDOWN")
    print("=" * 78)
    model_names = sorted(agg_df["model"].unique())
    col_w = 14
    header = f"  {'BA':<6}  {'EIA RMSE':>9}"
    for mn in model_names:
        short = mn[:col_w]
        header += f"  {short:>{col_w}}"
    header += f"  {'RMSE':>8}  {'vs EIA':>8}"
    # show per-model columns
    hdr2 = f"  {'':6}  {'':9}"
    for _ in model_names:
        hdr2 += f"  {'RMSE':>{col_w}}"
    print(header)
    print("-" * 78)

    pivot = agg_df.pivot(index="ba", columns="model", values=["rmse", "vs_eia_pct"])
    for ba in sorted(agg_df["ba"].unique()):
        ba_rows = agg_df[agg_df["ba"] == ba]
        eia_r   = ba_rows["eia_rmse"].iloc[0]
        row = f"  {ba:<6}  {eia_r:>9,.0f}"
        for mn in model_names:
            r = ba_rows[ba_rows["model"] == mn]
            if r.empty:
                row += f"  {'---':>{col_w}}"
            else:
                rmse = r["rmse"].iloc[0]
                pct  = r["vs_eia_pct"].iloc[0]
                beat = "B" if rmse < eia_r else " "
                sign = "+" if pct > 0 else ""
                if rmse > eia_r * 50:
                    cell = f"!!!EXPL {beat}"
                else:
                    cell = f"{rmse:,.0f}({sign}{pct:.0f}%){beat}"
                row += f"  {cell:>{col_w}}"
        print(row)

    if models_dir is not None:
        saved = list(models_dir.glob("*.joblib"))
        print(f"\nModels saved: {len(saved)} files in {models_dir}")

    # ── Feature importance top-10 per model ──────────────────────────────────
    _print_feature_importance(agg_df)
    
    # ── Final timing ──────────────────────────────────────────────────────────
    elapsed_total = time.perf_counter() - script_start_time
    print("\n" + "=" * 78)
    print(f"TRAINING COMPLETE — Total time: {_format_elapsed(elapsed_total)}")
    print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    return 0


def _print_feature_importance(agg_df: "pd.DataFrame") -> None:
    """Print top-10 features per model, averaged across all BAs."""
    import numpy as np
    from collections import defaultdict

    print("\n" + "=" * 78)
    print("TOP-10 FEATURE IMPORTANCE (averaged across all BAs, fold-4 model)")
    print("  *** = cv<0.3 consistent   ** = cv<0.6 moderate   * = varies by BA")
    print("=" * 78)

    for model_name, grp in agg_df.groupby("model"):
        # Collect importance vectors that are not None
        rows_with_imp = grp[grp["feature_importance"].notna()]
        if len(rows_with_imp) == 0:
            print(f"\n  {model_name}: no feature importance available")
            continue

        # Accumulate: feature -> list of importance values
        feat_vals = defaultdict(list)
        for _, row in rows_with_imp.iterrows():
            imp = row["feature_importance"]
            names = row["feature_names"]
            if imp is None or names is None or len(imp) != len(names):
                continue
            for fname, fval in zip(names, imp):
                feat_vals[fname].append(fval)

        if not feat_vals:
            continue

        means = {f: np.mean(v) for f, v in feat_vals.items()}
        stds  = {f: np.std(v)  for f, v in feat_vals.items()}
        n_bas = max(len(v) for v in feat_vals.values())
        ranked = sorted(means.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  {model_name}  ({n_bas} BAs)")
        print(f"  {'Rank':<5} {'Feature':<32} {'Mean %':>8}  {'Std %':>7}  {'CV'}")
        print(f"  {'-'*5} {'-'*32} {'-'*8}  {'-'*7}  {'-'*12}")
        for rank, (feat, mean_imp) in enumerate(ranked[:10], 1):
            pct     = mean_imp * 100
            std_pct = stds[feat] * 100
            cv      = stds[feat] / mean_imp if mean_imp > 0 else 99
            cons    = "***" if cv < 0.3 else ("** " if cv < 0.6 else "*  ")
            print(f"  #{rank:<4} {feat:<32} {pct:>7.2f}%  {std_pct:>6.2f}%  {cons}")


if __name__ == "__main__":
    sys.exit(main())
