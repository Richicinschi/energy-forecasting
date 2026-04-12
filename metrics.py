import re, sys

def parse_log(path, model_name):
    try:
        txt = open(path).read()
    except:
        return None
    beats = txt.count('BEAT')
    total = txt.count('vs EIA DF')
    eia_rmses = [int(m.replace(',','')) for m in re.findall(r'EIA DF RMSE=\s*([\d,]+)', txt)]
    model_rmses = [int(m.replace(',','')) for m in re.findall(model_name + r'\s+RMSE=\s*([\d,]+)', txt)]
    eia_mean = sum(eia_rmses)/len(eia_rmses) if eia_rmses else 0
    model_mean = sum(model_rmses)/len(model_rmses) if model_rmses else 0
    diff_pct = 100*(model_mean - eia_mean)/eia_mean if eia_mean else 0
    done = total == 51
    return {'beats': beats, 'total': total, 'eia_mean': eia_mean, 'model_mean': model_mean, 'diff_pct': diff_pct, 'done': done}

models = [
    ('XGBoost',       '/tmp/train_xgboost.log'),
    ('LightGBM',      '/tmp/train_lightgbm.log'),
    ('Ridge',         '/tmp/train_ridge.log'),
    ('HistGB',        '/tmp/train_hist_gb.log'),
    ('CatBoost',      '/tmp/train_catboost.log'),
    ('MLP',           '/tmp/train_mlp_all.log'),
    ('FTTransformer', '/tmp/train_ft_transformer_all.log'),
]

print(f"{'Model':<16} {'Status':<14} {'Wins':<12} {'Model RMSE':>12} {'EIA RMSE':>10} {'Diff':>10}")
print('-'*76)
for name, log in models:
    r = parse_log(log, name)
    if not r or r['total'] == 0:
        print(f"{name:<16} {'not started':<14}")
        continue
    status = 'DONE' if r['done'] else f"running {r['total']}/51"
    win_str = f"{r['beats']}/{r['total']} ({round(100*r['beats']/max(r['total'],1))}%)"
    sign = '+' if r['diff_pct'] > 0 else ''
    diff_str = sign + str(round(r['diff_pct'],1)) + '%'
    print(f"{name:<16} {status:<14} {win_str:<12} {round(r['model_mean']):>12,} MW {round(r['eia_mean']):>8,} MW {diff_str:>10}")
