# build_ml_table_from_sessions_clean_v2.py
import pandas as pd
from pathlib import Path
import numpy as np
from pandas.errors import PerformanceWarning
import warnings
warnings.simplefilter(action='ignore', category=PerformanceWarning)

BASE_DIR = Path("data/fastf1_sessions")
OUT_FILE_PQ = BASE_DIR.parent / "ml_table.parquet"
OUT_FILE_CSV = BASE_DIR.parent / "ml_table.csv"

# --- helper --------------------------------------------------------------
def safe_read_csv(path, required_cols):
    """
    Read CSV and return DataFrame with at least the required columns.
    If some required columns are missing, they will be created as NaNs.
    """
    df = pd.read_csv(path, low_memory=False)
    # Ensure lowercase column names for consistency
    df.columns = [c.lower() for c in df.columns]
    # Add any missing required columns as NaN
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# --- config --------------------------------------------------------------
REQ_QUAL_COLS = ['season', 'round', 'abbreviation', 'teamname', 'position']  # position = q pos
REQ_RACE_COLS  = ['season', 'round', 'abbreviation', 'teamname', 'position', 'points']

print("Gathering session files...")
qual_files = sorted(BASE_DIR.glob("*_Q.csv"))
race_files = sorted(BASE_DIR.glob("*_R.csv"))

if not qual_files or not race_files:
    raise FileNotFoundError("Could not find qualifying or race CSVs. Ensure the download script was run.")

print(f"Found {len(qual_files)} qualifying files and {len(race_files)} race files.")

# --- load & concat (robustly) -------------------------------------------
q_dfs = []
for f in qual_files:
    df = safe_read_csv(f, REQ_QUAL_COLS)
    # keep only required columns (lowercased)
    df = df[[c for c in REQ_QUAL_COLS if c in df.columns]]
    df.rename(columns={'position':'position'}, inplace=True)  # keep name until later
    q_dfs.append(df)

r_dfs = []
for f in race_files:
    df = safe_read_csv(f, REQ_RACE_COLS)
    df = df[[c for c in REQ_RACE_COLS if c in df.columns]]
    r_dfs.append(df)

q = pd.concat(q_dfs, ignore_index=True, sort=False)
r = pd.concat(r_dfs, ignore_index=True, sort=False)

# Normalize column names to lowercase
q.columns = [c.lower() for c in q.columns]
r.columns = [c.lower() for c in r.columns]

# Convert season/round to numeric if possible
q['season'] = pd.to_numeric(q.get('season', pd.Series(dtype='int64')), errors='coerce').astype('Int64')
q['round']  = pd.to_numeric(q.get('round', pd.Series(dtype='int64')), errors='coerce').astype('Int64')
r['season'] = pd.to_numeric(r.get('season', pd.Series(dtype='int64')), errors='coerce').astype('Int64')
r['round']  = pd.to_numeric(r.get('round', pd.Series(dtype='int64')), errors='coerce').astype('Int64')

# --- cleaning / rename -----------------------------------------------
q = q.rename(columns={'abbreviation':'driver', 'teamname':'constructor', 'position':'q_position'})
r = r.rename(columns={'abbreviation':'driver', 'teamname':'constructor', 'position':'race_position'})

# Convert numeric columns
q['q_position'] = pd.to_numeric(q['q_position'], errors='coerce')
r['race_position_num'] = pd.to_numeric(r['race_position'], errors='coerce')
if 'points' in r.columns:
    r['points'] = pd.to_numeric(r['points'], errors='coerce').fillna(0)
else:
    r['points'] = 0.0

# Keep essential columns (avoid carrying noisy extras)
q_small = q[['season','round','driver','constructor','q_position']].copy()
r_small = r[['season','round','driver','constructor','race_position_num','points']].copy()

# Convert driver/constructor to category after concat for memory savings
q_small['driver'] = q_small['driver'].astype('category')
q_small['constructor'] = q_small['constructor'].astype('category')
r_small['driver'] = r_small['driver'].astype('category')
r_small['constructor'] = r_small['constructor'].astype('category')

# --- merge and target -----------------------------------------------
df = pd.merge(
    r_small,
    q_small,
    on=['season','round','driver','constructor'],
    how='left',
    validate='m:1'  # many race rows -> max one qualific row per driver per race
)

# Target creation: podium (1 if finishing position <= 3)
df['podium'] = np.where(df['race_position_num'].le(3), 1, 0).astype('int8')

# --- rolling features (per driver) -------------------------------------
def add_rolling(g):
    g = g.sort_values(['season','round']).reset_index(drop=True)
    g['avg_pos_last3'] = g['race_position_num'].rolling(3, min_periods=1).mean().shift(1)
    g['podiums_last3']   = g['podium'].rolling(3, min_periods=1).sum().shift(1)
    g['points_last3']    = g['points'].rolling(3, min_periods=1).sum().shift(1)
    return g

df = df.groupby('driver', group_keys=False).apply(add_rolling)

# --- imputation --------------------------------------------------------
# q_position: fill missing with median of existing numeric q positions
median_q_pos = df['q_position'].median()
df['q_position'] = df['q_position'].fillna(median_q_pos)

# rolling features: initial NaNs -> 0 (no prior history)
df[['avg_pos_last3','podiums_last3','points_last3']] = df[['avg_pos_last3','podiums_last3','points_last3']].fillna(0)

# Final ML columns
df_ml = df[['season','round','driver','constructor',
            'q_position','avg_pos_last3','podiums_last3','points_last3',
            'podium','race_position_num','points']].copy()

# Convert categories to strings for storage (optional) or keep as categorical
df_ml['driver'] = df_ml['driver'].astype(str)
df_ml['constructor'] = df_ml['constructor'].astype(str)

# --- save --------------------------------------------------------------
df_ml.to_parquet(OUT_FILE_PQ, index=False)
df_ml.to_csv(OUT_FILE_CSV, index=False)
print(f"✅ Clean ML table saved: {OUT_FILE_PQ} and {OUT_FILE_CSV}  (shape={df_ml.shape})")

# --- quick sanity checks ----------------------------------------------
print("\nSanity checks:")
print("Seasons in dataset:", sorted(df_ml['season'].dropna().unique()))
print("Sample rows:\n", df_ml.head(6).to_string(index=False))
print("\nPodium distribution:\n", df_ml['podium'].value_counts(normalize=True).to_string())
