import pandas as pd
import numpy as np
from pathlib import Path

feat_dir = Path(__file__).parent / 'data' / 'processed' / 'features'
files = sorted(feat_dir.glob('*_features.parquet'))
print(f'Checking {len(files)} parquets...\n')

issues = []
for f in files:
    ba = f.stem.replace('_features', '')
    if ba == 'ALL':
        continue
    df = pd.read_parquet(f)
    d = df['demand_mw']
    eia = df['eia_forecast_mw']
    flags = []
    if (d > 500_000).any():   flags.append('OVERFLOW max=' + str(int(d.max())))
    if (d < 0).any():         flags.append('NEGATIVE ' + str(int((d<0).sum())) + ' rows')
    if (d == 0).any():        flags.append('ZERO ' + str(int((d==0).sum())) + ' rows')
    if (eia > 500_000).any(): flags.append('EIA_OVERFLOW max=' + str(int(eia.max())))
    if (eia < 0).any():       flags.append('EIA_NEG ' + str(int((eia<0).sum())) + ' rows')
    if (eia == 0).any():      flags.append('EIA_ZERO ' + str(int((eia==0).sum())) + ' rows')
    med = d.median()
    if (d > med * 10).any():  flags.append('OUTLIER >10x median max=' + str(int(d.max())) + ' med=' + str(int(med)))
    nan_d = d.isna().mean()
    if nan_d > 0.02:          flags.append('HIGH_NaN ' + str(round(nan_d*100,1)) + 'pct')
    if flags:
        issues.append(ba)
        print('[ISSUE] ' + ba + ': ' + ' | '.join(flags))
    else:
        print('[OK]    ' + ba + '  median=' + str(int(med)) + ' MW  max=' + str(int(d.max())) + ' MW  rows=' + str(len(df)))

print()
if issues:
    print('RESULT: ' + str(len(issues)) + ' BAs still have issues: ' + ', '.join(issues))
else:
    print('RESULT: ALL ' + str(len(files)) + ' BAs CLEAN - zero bad values')
