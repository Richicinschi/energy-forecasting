#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

# Same logic as train_models.py
project_root = Path('.')
features_dir = project_root / "data" / "processed" / "features"
path = features_dir / "MISO_features.parquet"

print(f"Reading: {path}")
print(f"Exists: {path.exists()}")

df = pd.read_parquet(path)
print(f"Total columns: {len(df.columns)}")
print(f"All columns: {list(df.columns)}")
print()

_EXCLUDE_COLS = {"demand_mw", "fold", "respondent", "is_imputed", "is_anomaly", "eia_forecast_mw"}
feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
print(f"Feature columns: {len(feature_cols)}")
print(f"Excluded: {[c for c in df.columns if c in _EXCLUDE_COLS]}")
print()
print(f"Sample feature cols: {feature_cols[:10]}")
