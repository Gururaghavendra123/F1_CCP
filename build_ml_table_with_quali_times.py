# build_ml_table_with_quali_times.py
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter(action='ignore', category=PerformanceWarning)

BASE_DIR = Path("data/fastf1_sessions")
OUT_PQ = BASE_DIR.parent / "ml_table_v2.parquet"
OUT_CSV = BASE_DIR.parent / "ml_table_v2.csv"

def safe_read_csv(path, required_cols):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# helper to parse FastF1 time strings like "1:29.708" or mm:ss.sss
def time_to_seconds(t):
    if pd.isna(t):
        return np.nan
    s = str(t).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            mins = float(parts[0])
            secs = float(parts[1])
            return mins*60 + secs
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

# collect files
qual_files = sorted(BASE_DIR.glob("*_Q.csv"))
race_files = sorted(BASE_DIR.glob("*_R.csv"))
if not qual_files or not race_files:
    raise FileNotFoundError("Missing Q or R csvs in data/fastf1_sessions")

# load qualifying and compute best q_time per driver per race
REQ_Q = ['season','round','abbreviation','teamname','q1','q2','q3','best']
q_rows = []
for f in qual_files:
    dfq = safe_read_csv(f, REQ_Q)
    # normalize names
    dfq = dfq.rename(columns={'abbreviation':'driver','teamname':'constructor'})
    # compute best time among Q1/Q2/Q3 or 'best' field if present
    for col in ['q1','q2','q3','best']:
        if col not in dfq.columns:
            dfq[col] = np.nan
    # convert to seconds
    for col in ['q1','q2','q3','best']:
        dfq[col + '_s'] = dfq[col].apply(time_to_seconds)
    # best available
    dfq['q_time_s'] = dfq[['q1_s','q2_s','q3_s','best_s']].min(axis=1)
    # keep essential columns
    q_rows.append(dfq[['season','round','driver','constructor','q_time_s','q1','q2','q3']])

q_all = pd.concat(q_rows, ignore_index=True, sort=False)
q_all['season'] = pd.to_numeric(q_all['season'], errors='coerce').astype('Int64')
q_all['round']  = pd.to_numeric(q_all['round'], errors='coerce').astype('Int64')

# load race files
REQ_R = ['season','round','abbreviation','teamname','position','points']
r_rows = []
for f in race_files:
    dfr = safe_read_csv(f, REQ_R)
    dfr = dfr.rename(columns={'abbreviation':'driver','teamname':'constructor','position':'race_position'})
    if 'points' not in dfr.columns:
        dfr['points'] = 0.0
    dfr['race_position_num'] = pd.to_numeric(dfr['race_position'], errors='coerce')
    r_rows.append(dfr[['season','round','driver','constructor','race_position_num','points']])

r_all = pd.concat(r_rows, ignore_index=True, sort=False)
r_all['season'] = pd.to_numeric(r_all['season'], errors='coerce').astype('Int64')
r_all['round']  = pd.to_numeric(r_all['round'], errors='coerce').astype('Int64')

# merge race + quali times
df = pd.merge(r_all, q_all[['season','round','driver','q_time_s']], on=['season','round','driver'], how='left')

# target
df['podium'] = np.where(df['race_position_num'].le(3), 1, 0).astype('int8')

# rolling features (same as before)
def add_rolling(g):
    g = g.sort_values(['season','round']).reset_index(drop=True)
    g['avg_pos_last3'] = g['race_position_num'].rolling(3, min_periods=1).mean().shift(1)
    g['podiums_last3'] = g['podium'].rolling(3, min_periods=1).sum().shift(1)
    g['points_last3'] = g['points'].rolling(3, min_periods=1).sum().shift(1)
    return g

df = df.groupby('driver', group_keys=False).apply(add_rolling)

# fill naive missing values
median_q_time = df['q_time_s'].median()
df['q_time_s'] = df['q_time_s'].fillna(median_q_time)
df[['avg_pos_last3','podiums_last3','points_last3']] = df[['avg_pos_last3','podiums_last3','points_last3']].fillna(0)

# --- compute teammate qualifying delta ---
# find teammate q_time per season+round+constructor
df['constructor'] = df['constructor'].astype(str)
# join to get teammate best times (group by season,round,constructor)
team_q = df.groupby(['season','round','constructor'])['q_time_s'].apply(list).reset_index().rename(columns={'q_time_s':'team_q_times'})
# join back
df = df.merge(team_q, on=['season','round','constructor'], how='left')
# compute teammate delta: driver's q_time minus min teammate other than self (if two drivers)
def teammate_delta_row(row):
    times = row['team_q_times']
    if not isinstance(times, list) or len(times) <= 1:
        return 0.0  # no teammate or only one time -> neutral
    try:
        my = row['q_time_s']
        # remove my if duplicate values; safer approach: compute min of other times if possible
        others = [t for t in times if not np.isclose(t, my, equal_nan=True)]
        if not others:
            # if all equal or couldn't remove, compare to min (may be same)
            others = times
        delta = my - np.min(others)
        return float(delta)
    except:
        return 0.0

df['teammate_delta_q'] = df.apply(teammate_delta_row, axis=1)

# cleanup
df = df.drop(columns=['team_q_times'])

# final selection and save
df_ml = df[['season','round','driver','constructor','q_time_s','teammate_delta_q',
            'avg_pos_last3','podiums_last3','points_last3','podium','race_position_num','points']].copy()

df_ml.to_parquet(OUT_PQ, index=False)
df_ml.to_csv(OUT_CSV, index=False)
print("Saved ML table v2:", OUT_PQ, df_ml.shape)
