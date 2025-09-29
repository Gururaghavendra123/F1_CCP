# rebuild_ml_v3_normalized_q.py  (robust fixed version)
import pandas as pd
import numpy as np
from pathlib import Path

BASE = "data/fastf1_sessions"
OUT = "data/ml_table_v3.parquet"

def time_to_s(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "n/a", "-"):
        return np.nan
    s = s.replace(",", ".")
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        secs = float(parts[-1])
        total = secs
        # accumulate minutes/hours if present
        for p in range(len(parts)-2, -1, -1):
            try:
                val = float(parts[p])
            except:
                try:
                    import re
                    dig = re.findall(r"[\d\.]+", parts[p])
                    val = float(dig[-1]) if dig else 0.0
                except:
                    val = 0.0
            total += val * (60 ** (len(parts)-1-p))
        return float(total)
    except Exception:
        try:
            import re
            dig = re.findall(r"[\d]+\.[\d]+|[\d]+", s)
            if not dig:
                return np.nan
            return float(dig[0])
        except:
            return np.nan

# gather files
q_files = sorted(Path(BASE).glob("*_Q.csv"))
r_files = sorted(Path(BASE).glob("*_R.csv"))

if not q_files or not r_files:
    raise FileNotFoundError("No qualifying or race CSVs found in data/fastf1_sessions")

# build qualifying DF
q_list = []
for f in q_files:
    df = pd.read_csv(f, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    # ensure expected columns exist
    for c in ['q1','q2','q3','best','abbreviation','teamname','season','round']:
        if c not in df.columns:
            df[c] = np.nan
    df = df.rename(columns={'abbreviation':'driver','teamname':'constructor'})
    for col in ['q1','q2','q3','best']:
        df[col + '_s'] = df[col].apply(time_to_s)
    df['q_time_s'] = df[['q1_s','q2_s','q3_s','best_s']].min(axis=1)
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df['round'] = pd.to_numeric(df['round'], errors='coerce')
    q_list.append(df[['season','round','driver','constructor','q_time_s']])

q_all = pd.concat(q_list, ignore_index=True, sort=False)

# build race DF
r_list = []
for f in r_files:
    df = pd.read_csv(f, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    for c in ['abbreviation','teamname','position','points','season','round']:
        if c not in df.columns:
            df[c] = np.nan
    df = df.rename(columns={'abbreviation':'driver','teamname':'constructor','position':'race_position'})
    df['race_position_num'] = pd.to_numeric(df['race_position'], errors='coerce')
    df['points'] = pd.to_numeric(df.get('points', 0), errors='coerce').fillna(0)
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df['round'] = pd.to_numeric(df['round'], errors='coerce')
    r_list.append(df[['season','round','driver','constructor','race_position_num','points']])

r_all = pd.concat(r_list, ignore_index=True, sort=False)

# merge race + qual
df = pd.merge(r_all, q_all, on=['season','round','driver'], how='left')

# ensure constructor exists and fill missing with "UNKNOWN"
if 'constructor' not in df.columns:
    df['constructor'] = "UNKNOWN"
else:
    df['constructor'] = df['constructor'].astype(str).fillna('UNKNOWN')

# compute session-normalized qualifying delta (q_delta_s)
df['q_time_s'] = pd.to_numeric(df['q_time_s'], errors='coerce')
df['q_min'] = df.groupby(['season','round'])['q_time_s'].transform('min')
df['q_delta_s'] = df['q_time_s'] - df['q_min']

# create target and basic fields
df['podium'] = (df['race_position_num'] <= 3).astype(int)
df['points'] = pd.to_numeric(df['points'], errors='coerce').fillna(0)

# rolling (per-driver) - select only non-grouping columns in apply to silence future warning
def add_roll(g):
    g = g.sort_values(['season','round']).reset_index(drop=True)
    g['avg_pos_last3'] = g['race_position_num'].rolling(3, min_periods=1).mean().shift(1)
    g['podiums_last3'] = g['podium'].rolling(3, min_periods=1).sum().shift(1)
    g['points_last3'] = g['points'].rolling(3, min_periods=1).sum().shift(1)
    return g

df = df.groupby('driver', group_keys=False).apply(lambda g: add_roll(g.drop(columns=[])) )

# teammate delta: compute team minimum q_delta per session, then subtract
# ensure q_delta_s exists (it may be NaN for sessions w/ missing qual data)
df['q_delta_s'] = pd.to_numeric(df['q_delta_s'], errors='coerce')
# compute team min safely
df['team_min_qdelta'] = df.groupby(['season','round','constructor'])['q_delta_s'].transform('min')
# compute teammate delta and fill NaNs with 0
df['teammate_delta_q2'] = (df['q_delta_s'] - df['team_min_qdelta']).fillna(0.0)

# final imputation: fill q_delta median for missing, rolling features with 0
median_qd = df['q_delta_s'].median(skipna=True)
df['q_delta_s'] = df['q_delta_s'].fillna(median_qd)
df[['avg_pos_last3','podiums_last3','points_last3']] = df[['avg_pos_last3','podiums_last3','points_last3']].fillna(0)

# select & save
out_cols = ['season','round','driver','constructor','q_delta_s','teammate_delta_q2',
            'avg_pos_last3','podiums_last3','points_last3','podium','race_position_num','points']
df_ml3 = df[out_cols].copy()
Path("data").mkdir(parents=True, exist_ok=True)
df_ml3.to_parquet(OUT, index=False)
print("Created", OUT, "shape:", df_ml3.shape)
