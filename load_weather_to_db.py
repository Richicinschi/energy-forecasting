#!/usr/bin/env python3
"""Load cached weather parquet files into SQLite database."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from src.data.database import get_engine, create_all_tables

def save_weather_to_db_chunked(df: pd.DataFrame, engine, chunk_size: int = 100) -> int:
    """Save weather DataFrame to SQLite in chunks to avoid parameter limits."""
    total_rows = 0
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        chunk.to_sql(
            'weather_data',
            engine,
            if_exists='append',
            index=False,
            method='multi'
        )
        total_rows += len(chunk)
    return total_rows

def main():
    engine = get_engine()
    create_all_tables(engine)
    
    # Clear existing weather data to avoid UNIQUE constraint errors
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM weather_data"))
        count = result.scalar()
        print(f'Clearing {count:,} existing rows from weather_data')
        conn.execute(text("DELETE FROM weather_data"))
        conn.commit()

    cache_dir = Path('data/raw/weather')
    parquet_files = sorted(cache_dir.glob('*.parquet'))
    print(f'Found {len(parquet_files)} parquet files')

    total_rows = 0
    for i, f in enumerate(parquet_files, 1):
        df = pd.read_parquet(f)
        
        # Reset index to get 'period' as a column
        if df.index.name == 'period':
            df = df.reset_index()
        elif 'period' not in df.columns:
            # If index is unnamed but is datetime, use it as period
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df = df.rename(columns={'index': 'period'})
        
        # Ensure period is datetime
        if 'period' in df.columns:
            df['period'] = pd.to_datetime(df['period'], utc=True)
        else:
            print(f"  Warning: {f.stem} has no period column, skipping")
            continue
        
        # Add respondent if missing
        if 'respondent' not in df.columns:
            df['respondent'] = f.stem
        
        # Ensure all expected columns exist (6 core columns from API)
        expected_cols = ['period', 'respondent', 'temp_2m', 'dewpoint_2m', 
                        'windspeed_10m', 'solar_irradiance', 'cloudcover', 'precipitation']
        for col in expected_cols:
            if col not in df.columns:
                print(f"  Warning: {f.stem} missing column {col}")
                df[col] = None
        
        # Reorder columns to match table
        df = df[expected_cols]
        
        rows = save_weather_to_db_chunked(df, engine, chunk_size=100)
        total_rows += rows
        print(f'[{i}/{len(parquet_files)}] {f.stem}: {rows:,} rows')

    print(f'\nTotal rows inserted: {total_rows:,}')
    return 0

if __name__ == "__main__":
    from src.data.database import get_engine
    sys.exit(main())
