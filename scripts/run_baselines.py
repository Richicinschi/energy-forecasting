#!/usr/bin/env python3
"""
run_baselines.py — Evaluate all three baseline models via walk-forward CV.

Reads per-BA Parquet feature files and evaluates:
  - Persistence (lag 1h)
  - Seasonal Naive (lag 168h)
  - EIA Day-Ahead Forecast (DF column)

Results are printed as a comparison table and saved to
data/processed/baseline_results.csv.

Usage:
    # Single BA
    python scripts/run_baselines.py --ba MISO

    # Multiple BAs
    python scripts/run_baselines.py --ba MISO PJM ERCO

    # All BAs with feature files
    python scripts/run_baselines.py --ba ALL

    # Custom features dir or output path
    python scripts/run_baselines.py --ba MISO --features-dir data/processed/features/ --output data/processed/baseline_results.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.evaluation.metrics import compare_models, summary_table
from src.models.baselines import (
    EIAForecastModel,
    Persistence1hModel,
    Persistence24hModel,
    SeasonalNaiveModel,
)

# Ordered for display: 1h first (upper bound), then 24h-horizon group
BASELINES = [
    Persistence1hModel(),
    Persistence24hModel(),
    SeasonalNaiveModel(),
    EIAForecastModel(),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate baseline models on energy demand feature matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_baselines.py --ba MISO
  python scripts/run_baselines.py --ba MISO PJM ERCO
  python scripts/run_baselines.py --ba ALL
        """,
    )
    parser.add_argument(
        "--ba", nargs="+", default=["ALL"], metavar="CODE",
        help="BA codes to evaluate, or ALL (default: ALL)",
    )
    parser.add_argument(
        "--features-dir", default=None, metavar="PATH",
        help="Directory containing {BA}_features.parquet files",
    )
    parser.add_argument(
        "--output", default=None, metavar="PATH",
        help="CSV output path (default: data/processed/baseline_results.csv)",
    )
    parser.add_argument(
        "--no-per-fold", action="store_true",
        help="Only print the overall summary, not per-fold breakdown",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    features_dir = (
        Path(args.features_dir) if args.features_dir
        else project_root / "data" / "processed" / "features"
    )
    output_path = (
        Path(args.output) if args.output
        else project_root / "data" / "processed" / "baseline_results.csv"
    )

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

    print(f"\nBaseline evaluation — {len(ba_codes)} BA(s)")
    print("=" * 70)

    all_results = []

    for ba in ba_codes:
        path = features_dir / f"{ba}_features.parquet"
        if not path.exists():
            print(f"  {ba}: no feature file found — skipping")
            continue

        df = pd.read_parquet(path)

        # Run all baselines
        results = compare_models(BASELINES, df)
        results.insert(1, "ba", ba)
        all_results.append(results)

        # Print per-BA summary
        overall = summary_table(results)
        print(f"\n  {ba}  ({len(df):,} rows)")
        print(overall[["model", "mae", "rmse", "smape", "n_rows"]].to_string(index=True))
        print()
        print("  NOTE: EIA DF and Persistence 24h/168h are all 24h+-ahead forecasts.")
        print("  Persistence 1h is included as an upper-bound reference only (1h horizon).")

        if not args.no_per_fold:
            # Show per-fold RMSE for EIA and Persistence24h (the key comparison)
            for model_name in [EIAForecastModel.name, Persistence24hModel.name]:
                fold_rows = results[
                    (results["model"] == model_name)
                    & (results["fold"] != "overall")
                ]
                if not fold_rows.empty:
                    fold_rmse = fold_rows[["fold", "rmse"]].set_index("fold")["rmse"].round(0).astype(int).to_dict()
                    print(f"    {model_name} per-fold RMSE: {fold_rmse}")

    if not all_results:
        print("\nNo results produced.")
        return 1

    # Save combined results
    combined = pd.concat(all_results, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    # Print aggregate summary across all BAs (overall rows only)
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY (mean across all BAs, overall fold)")
    print("=" * 70)
    overall_rows = combined[combined["fold"] == "overall"].copy()
    agg = (
        overall_rows.groupby("model")[["mae", "rmse", "smape"]]
        .mean()
        .round(2)
        .sort_values("rmse")
    )
    print(agg.to_string())
    print(f"\nResults saved: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
