#!/usr/bin/env python3
"""Phase 4: Cross-BA Spatial Features

Captures regional demand patterns, neighbor effects, and spatial weather correlations.
All features are leakage-safe (use t-24 data only).
"""

import numpy as np
import pandas as pd
from sqlalchemy import text


# Define BA regions for macro patterns
BA_REGIONS = {
    # Eastern Interconnection
    "eastern": ["PJM", "MISO", "ISNE", "NYIS", "TVA", "SOCO", "DUK", "CPLE", "CPLW", 
                "AEC", "FMPP", "FPC", "FPL", "GVL", "JEA", "SEC", "SC", "SCEG", 
                "TAL", "TEC", "LGEE", "SWPP"],
    # Western Interconnection  
    "western": ["CISO", "BPAT", "BANC", "CHPD", "DOPD", "GCPD", "NWMT", "AVA",
                "IPCO", "PACE", "PACW", "PGE", "PSEI", "SCL", "SNM", "WACM", "WAUW",
                "LDWP", "NEVP", "SRP", "TEPC", "WALC", "IID"],
    # Texas (ERCOT)
    "texas": ["ERCO", "EPE"],
    # Additional mapping for specific neighbors
}

# Define adjacent/neighboring BAs (based on geography/interchange patterns)
NEIGHBOR_MAP = {
    "PJM": ["MISO", "NYIS", "TVA", "ISNE"],
    "MISO": ["PJM", "SWPP", "SOCO", "TVA"],
    "ISNE": ["NYIS", "PJM"],
    "NYIS": ["ISNE", "PJM"],
    "TVA": ["PJM", "MISO", "SOCO"],
    "SOCO": ["TVA", "MISO", "FPL"],
    "ERCO": ["SWPP", "EPE"],
    "CISO": ["BPAT", "BANC", "LDWP"],
    "BPAT": ["CISO", "CHPD", "DOPD", "GCPD"],
    # Add more as needed...
}


def add_spatial_features(df: pd.DataFrame, engine, respondent: str, all_ba_codes: list) -> pd.DataFrame:
    """Add Phase 4 cross-BA spatial features.
    
    Args:
        df: DataFrame with UTC DatetimeIndex for target BA
        engine: SQLAlchemy engine
        respondent: Target BA code
        all_ba_codes: List of all BA codes in dataset
    
    Returns:
        DataFrame with spatial features added
    """
    idx = df.index
    
    # ── Regional demand index ────────────────────────────────────────────────
    # Sum of demand in same region (Eastern/Western/Texas)
    region = None
    for reg_name, bas in BA_REGIONS.items():
        if respondent in bas:
            region = reg_name
            break
    
    if region and region != "texas":  # Texas only has ERCO
        other_bas = [ba for ba in BA_REGIONS[region] if ba != respondent and ba in all_ba_codes]
        if other_bas:
            region_demand = _get_regional_demand(engine, other_bas, idx)
            if region_demand is not None:
                df["regional_demand_index"] = region_demand.astype("float32")
                # Regional demand per capita (normalized)
                df["regional_demand_per_ba"] = (region_demand / len(other_bas)).astype("float32")
    
    # ── Neighbor BA features ─────────────────────────────────────────────────
    neighbors = NEIGHBOR_MAP.get(respondent, [])
    neighbors = [n for n in neighbors if n in all_ba_codes]
    
    if neighbors:
        neighbor_stats = _get_neighbor_stats(engine, neighbors, idx)
        if neighbor_stats is not None:
            df["neighbor_demand_avg"] = neighbor_stats["avg"].astype("float32")
            df["neighbor_demand_max"] = neighbor_stats["max"].astype("float32")
            df["neighbor_demand_min"] = neighbor_stats["min"].astype("float32")
            
            # Demand delta vs neighbors (is this BA an outlier?)
            if "lag_24h" in df.columns:
                df["demand_delta_vs_neighbors"] = (df["lag_24h"] - neighbor_stats["avg"]).astype("float32")
    
    # ── Weather delta vs neighbors ───────────────────────────────────────────
    if "temp_2m" in df.columns:
        neighbor_temps = _get_neighbor_weather(engine, neighbors, idx, "temp_2m")
        if neighbor_temps is not None:
            # Is this BA hotter/colder than neighbors?
            df["temp_delta_vs_neighbors"] = (df["temp_2m"] - neighbor_temps).astype("float32")
    
    # ── Interchange flow features (enhanced) ─────────────────────────────────
    # Already have interchange_net_lag24, add directional features
    if "interchange_net_lag24" in df.columns:
        net = df["interchange_net_lag24"]
        # Import stress (high positive = heavy imports)
        df["is_import_stress"] = (net > net.quantile(0.9)).astype("int8")
        # Export stress (high negative = heavy exports)  
        df["is_export_stress"] = (net < net.quantile(0.1)).astype("int8")
    
    # ── Grid stress index ────────────────────────────────────────────────────
    # Composite of regional demand, neighbor stress, and interchange
    stress_components = []
    if "regional_demand_index" in df.columns:
        stress_components.append(df["regional_demand_index"] / df["regional_demand_index"].max())
    if "neighbor_demand_avg" in df.columns:
        stress_components.append(df["neighbor_demand_avg"] / df["neighbor_demand_avg"].max())
    
    if stress_components:
        df["grid_stress_index"] = np.mean(stress_components, axis=0).astype("float32")
    
    return df


def _get_regional_demand(engine, ba_codes: list, idx: pd.DatetimeIndex) -> pd.Series | None:
    """Get sum of demand for a list of BAs (regional aggregate)."""
    try:
        # Query demand (D) for all BAs in region
        ba_list = "', '".join(ba_codes)
        q = text(f"""
            SELECT period, respondent, value_mwh as demand
            FROM region_data
            WHERE respondent IN ('{ba_list}') AND type = 'D'
            AND period BETWEEN :start AND :end
            ORDER BY period
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql_query(q, conn, params={
                "start": idx.min(), "end": idx.max()
            }, parse_dates=["period"])
        
        if df.empty:
            return None
        
        # Pivot and sum
        df = df.pivot(index="period", columns="respondent", values="demand")
        df = df.reindex(idx)
        # Shift 24h for leakage safety
        df = df.shift(24)
        # Sum across BAs (handle NaN)
        return df.sum(axis=1, skipna=True)
    except Exception:
        return None


def _get_neighbor_stats(engine, ba_codes: list, idx: pd.DatetimeIndex) -> dict | None:
    """Get demand statistics for neighbor BAs."""
    try:
        ba_list = "', '".join(ba_codes)
        q = text(f"""
            SELECT period, respondent, value_mwh as demand
            FROM region_data
            WHERE respondent IN ('{ba_list}') AND type = 'D'
            AND period BETWEEN :start AND :end
            ORDER BY period
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql_query(q, conn, params={
                "start": idx.min(), "end": idx.max()
            }, parse_dates=["period"])
        
        if df.empty:
            return None
        
        df = df.pivot(index="period", columns="respondent", values="demand")
        df = df.reindex(idx)
        df = df.shift(24)  # Leakage safety
        
        return {
            "avg": df.mean(axis=1, skipna=True),
            "max": df.max(axis=1, skipna=True),
            "min": df.min(axis=1, skipna=True),
        }
    except Exception:
        return None


def _get_neighbor_weather(engine, ba_codes: list, idx: pd.DatetimeIndex, field: str) -> pd.Series | None:
    """Get average weather field for neighbor BAs."""
    try:
        ba_list = "', '".join(ba_codes)
        q = text(f"""
            SELECT period, respondent, {field} as value
            FROM weather_data
            WHERE respondent IN ('{ba_list}')
            AND period BETWEEN :start AND :end
            ORDER BY period
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql_query(q, conn, params={
                "start": idx.min(), "end": idx.max()
            }, parse_dates=["period"])
        
        if df.empty:
            return None
        
        df = df.pivot(index="period", columns="respondent", values="value")
        df = df.reindex(idx)
        df = df.shift(24)  # Leakage safety
        return df.mean(axis=1, skipna=True)
    except Exception:
        return None
