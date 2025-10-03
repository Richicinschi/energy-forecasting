# Energy Load Forecasting

A production-grade machine learning pipeline for hourly electricity demand forecasting using EIA-930 data.

## Project Goal

Forecast hourly electricity demand for **all EIA-930 Balancing Authorities (BAs)** and beat EIA's own published demand forecast (`DF`) on every region — individually and in aggregate.

## What Makes This Different

- Fetches **both** `D` (actual demand) and `DF` (EIA's official forecast) from the EIA Open Data API
- Benchmarks every ML model against EIA's own forecast — beating it is the explicit success criterion
- Covers **all ~65 EIA Balancing Authorities** individually, then combined
- Production-oriented: drift detection, automated retraining, MLflow experiment tracking
- Interactive Streamlit dashboard for stakeholder exploration

## Architecture

```
energy-forecasting/
├── config/
│   └── config.yaml          # All regions, model params, drift thresholds
├── src/
│   ├── data/
│   │   ├── eia_client.py    # EIA Open Data API v2 client (paginated)
│   │   ├── ingest.py        # Validation, gap filling, SQLite storage
│   │   └── database.py      # SQLAlchemy models
│   ├── features/
│   │   ├── preprocess.py    # Time indexing, anomaly flagging
│   │   └── engineer.py      # Lag, rolling, calendar, Fourier features
│   ├── models/
│   │   ├── baselines.py     # Persistence, seasonal naive, EIA-DF wrapper
│   │   └── ml_models.py     # Ridge, RandomForest, HistGradientBoosting
│   ├── evaluation/
│   │   ├── metrics.py       # MAE, RMSE, sMAPE
│   │   └── evaluate.py      # Walk-forward CV, model comparison
│   ├── monitoring/
│   │   ├── drift.py         # Sliding-window RMSE drift detection + Evidently
│   │   └── retrain.py       # Triggered retraining + MLflow registry
│   └── visualization/
│       └── plots.py         # Plotly + matplotlib chart functions
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_comparison.ipynb  # The "beat EIA" notebook
│   └── 04_visualizations.ipynb
├── scripts/                 # CLI entry points for each pipeline stage
├── tests/                   # pytest suite
├── data/
│   ├── raw/                 # Raw API downloads (gitignored)
│   └── processed/           # Cleaned parquet + evaluation CSVs (gitignored)
├── models/saved/            # Serialized model files (gitignored)
├── logs/                    # Drift logs (gitignored)
└── mlruns/                  # MLflow experiments (gitignored)
```

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo>
cd energy-forecasting
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure your EIA API key (free at https://www.eia.gov/opendata/)
cp env.example .env
# Edit .env and add your EIA_API_KEY

# 3. Verify setup
python verify_setup.py

# 4. Fetch data for a single region
python scripts/fetch_data.py --region MISO --start 2022-01-01 --end 2023-12-31

# 5. Fetch data for ALL balancing authorities
python scripts/fetch_data.py --region ALL --start 2022-01-01 --end 2023-12-31

# 6. Run full pipeline
python scripts/run_pipeline.py --region MISO

# 7. Launch dashboard
streamlit run dashboard/app.py
```

## EIA-930 Balancing Authorities

This project targets all ~65 EIA Balancing Authorities. Key regions:

| BA Code | Name | Notes |
|---------|------|-------|
| ERCO | ERCOT | Texas grid, largest standalone BA |
| MISO | Midcontinent ISO | Largest by area |
| PJM | PJM Interconnection | Largest by load |
| SWPP | Southwest Power Pool | |
| CISO | California ISO (CAISO) | |
| NYIS | New York ISO | |
| ISNE | ISO New England | |
| SOCO | Southern Company | |
| TVA | Tennessee Valley Authority | |
| BPAT | Bonneville Power Administration | |

See `config/config.yaml` for the complete list of all supported BAs.

## Models

| Model | Type | Notes |
|-------|------|-------|
| Persistence | Baseline | demand(t) = demand(t-1) |
| Seasonal Naive | Baseline | demand(t) = demand(t-168h) |
| EIA DF | Baseline | **EIA's own official forecast — the bar to beat** |
| Ridge Regression | ML | Linear with L2 regularization |
| Random Forest | ML | 200 estimators |
| HistGradientBoosting | ML | Native missing-value support |

## Evaluation

All models evaluated on a **chronological 80/20 holdout** (no shuffling) using:
- **MAE** — Mean Absolute Error (MWh)
- **RMSE** — Root Mean Squared Error (MWh)  
- **sMAPE** — Symmetric Mean Absolute Percentage Error (%)

Walk-forward TimeSeriesSplit cross-validation (5 folds) for model selection.

## Features (~25 total)

- **Calendar**: hour_of_day, day_of_week, month, quarter, is_weekend, is_holiday
- **Lag**: demand_lag_1h, _2h, _24h, _48h, _168h (1 week)
- **Rolling**: rolling_mean_24h, rolling_std_24h, rolling_mean_168h, rolling_std_168h
- **Fourier**: sin/cos pairs for daily seasonality (k=2) and weekly seasonality (k=3)

## Production Features

- **Drift detection**: Sliding 7-day RMSE window vs training baseline; alerts at >15% degradation
- **Auto-retraining**: Triggered on drift; champion model replaced only if metrics improve
- **Experiment tracking**: MLflow for all runs, parameters, metrics, and model artifacts
- **Evidently reports**: HTML data drift and model performance reports

## Repository

This project is designed to be a portfolio piece demonstrating:
- Real API data ingestion with pagination and validation
- Professional time-series feature engineering
- Rigorous model evaluation (no data leakage)
- Production-grade monitoring and retraining
- Clean, reproducible code structure

---

*Data source: EIA Open Data API v2 — https://www.eia.gov/opendata/*
