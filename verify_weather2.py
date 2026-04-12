#!/usr/bin/env python3
import pandas as pd
from src.data.database import get_engine
from sqlalchemy import text

engine = get_engine()

# Check distinct timestamps
print("Sample of available timestamps for ERCO:")
with engine.connect() as conn:
    q = text("""SELECT period, temp_2m 
                FROM weather_data 
                WHERE respondent = 'ERCO' 
                LIMIT 5""")
    result = conn.execute(q)
    for row in result:
        print(f"  {row[0]}: {row[1]}C")

print()

# Compare temperature ranges across BAs
print("Temperature Statistics by BA (2022):")
bas = ['ERCO', 'ISNE', 'MISO', 'CISO', 'PJM', 'SWPP']
with engine.connect() as conn:
    for ba in bas:
        q = text("""SELECT 
                      MIN(temp_2m) as min_temp, 
                      MAX(temp_2m) as max_temp,
                      AVG(temp_2m) as avg_temp
                    FROM weather_data 
                    WHERE respondent = :ba 
                    AND period >= '2022-01-01' 
                    AND period < '2023-01-01'""")
        result = conn.execute(q, {'ba': ba})
        row = result.fetchone()
        print(f"  {ba}: min={row[0]:5.1f}C, max={row[1]:5.1f}C, avg={row[2]:5.1f}C")

print()

# Check if temperatures make sense geographically
print("Geographic Sanity Check (colder in north, warmer in south):")
print("  ISNE (42.5N) should be coldest")
print("  ERCO (31.0N) should be warmest")
with engine.connect() as conn:
    q = text("""SELECT respondent, AVG(temp_2m) as avg_temp
                FROM weather_data 
                WHERE period >= '2022-01-01' AND period < '2023-01-01'
                AND respondent IN ('ERCO', 'ISNE', 'MISO', 'CISO')
                GROUP BY respondent
                ORDER BY avg_temp""")
    result = conn.execute(q)
    print("\n  Ranked by avg temp (coldest first):")
    for row in result:
        print(f"    {row[0]}: {row[1]:.1f}C")
