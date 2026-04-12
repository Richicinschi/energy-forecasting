#!/usr/bin/env python3
"""Benchmark Pandas vs Polars for Parquet reading."""

import time
import pandas as pd
import polars as pl

print("Benchmarking Parquet read speed...")
print("=" * 50)

# Pandas
start = time.time()
df_pd = pd.read_parquet('data/processed/features/MISO_features.parquet')
pd_time = time.time() - start
print(f'Pandas: {pd_time:.3f}s, shape: {df_pd.shape}')

# Polars
start = time.time()
df_pl = pl.read_parquet('data/processed/features/MISO_features.parquet')
pl_time = time.time() - start
print(f'Polars: {pl_time:.3f}s, shape: {df_pl.shape}')

print("=" * 50)
print(f'Speedup: {pd_time/pl_time:.1f}x')

if pd_time > pl_time:
    print(f"Polars is {pd_time/pl_time:.1f}x faster!")
else:
    print(f"Pandas is {pl_time/pd_time:.1f}x faster!")
