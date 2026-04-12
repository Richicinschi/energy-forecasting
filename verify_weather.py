#!/usr/bin/env python3
import pandas as pd
from src.data.ba_coordinates import get_ba_coordinates
from src.data.database import get_engine
from sqlalchemy import text

# Check coordinates for key BAs
bas = ['ERCO', 'ISNE', 'MISO', 'CISO', 'PJM', 'SWPP']
print("BA Coordinates:")
for ba in bas:
    coord = get_ba_coordinates(ba)
    print(f"  {ba}: lat={coord['lat']}, lon={coord['lon']}")

print()

# Check weather data differences
print("Sample Weather Data (Jan 15, 2022 12:00):")
engine = get_engine()
with engine.connect() as conn:
    for ba in bas:
        q = text("""SELECT temp_2m, dewpoint_2m, windspeed_10m, solar_irradiance 
                  FROM weather_data 
                  WHERE respondent = :ba AND period = '2022-01-15 12:00:00+00:00'""")
        result = conn.execute(q, {'ba': ba})
        row = result.fetchone()
        if row:
            print(f"  {ba}: temp={row[0]:5.1f}C, dew={row[1]:5.1f}C, wind={row[2]:4.1f}, solar={row[3]:6.1f}")
        else:
            print(f"  {ba}: NO DATA")

print()

# Check summer vs winter for ERCO (Texas)
print("ERCO Temperature Range (2022):")
with engine.connect() as conn:
    q = text("""SELECT 
                  MIN(temp_2m) as min_temp, 
                  MAX(temp_2m) as max_temp,
                  AVG(temp_2m) as avg_temp
                FROM weather_data 
                WHERE respondent = 'ERCO' 
                AND period >= '2022-01-01' 
                AND period < '2023-01-01'""")
    result = conn.execute(q)
    row = result.fetchone()
    print(f"  Min: {row[0]:.1f}C, Max: {row[1]:.1f}C, Avg: {row[2]:.1f}C")

# Compare ISNE vs ERCO summer temps
print()
print("July 2022 Average Temperatures:")
with engine.connect() as conn:
    for ba in ['ERCO', 'ISNE']:
        q = text("""SELECT AVG(temp_2m) as avg_temp
                      FROM weather_data 
                      WHERE respondent = :ba 
                      AND period >= '2022-07-01' 
                      AND period < '2022-08-01'""")
        result = conn.execute(q, {'ba': ba})
        row = result.fetchone()
        print(f"  {ba}: {row[0]:.1f}C")
