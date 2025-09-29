# predict_with_ensemble_and_save.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import fastf1

fastf1.Cache.enable_cache("f1_cache")

def time_to_seconds(t):
    if pd.isna(t):
        return np.nan
    s = str(t).strip().replace(',', '.')
    parts = s.split(':')
    try:
        if len(parts) == 1:
            return float(parts[0])
        total = float(parts[-1])
        for p in range(len(parts)-2, -1, -1):
            total += float(parts[p]) * (60 ** (len(parts)-1-p))
        return float(total)
    except:
        import re
        m = re.findall(r"[\d]+\.[\d]+|[\d]+", s)
        return float(m[0]) if m else np.nan

def fetch_session_csv(season:int, rnd:int, prefer='Q'):
    outdir = Path("data/fastf1_sessions"); outdir.mkdir(parents=True, exist_ok=True)
    for s_type in ([prefer] if prefer else []) + (['R'] if prefer=='Q' else ['Q']):
        fname = outdir / f"{season}_round{rnd:02d}_{s_type}.csv"
        if fname.exists(): return str(fname)
        try:
            session = fastf1.get_session(int(season), int(rnd), s_type)
        except Exception:
            continue
        try:
            session.load(laps=False, telemetry=False)
        except Exception:
            pass
        if session.results is None or session.results.empty:
            continue
        df = session.results.reset_index(drop=True).copy()
        df.columns = [c for c in df.columns]
        df.to_csv(fname, index=False)
        return str(fname)
    return None

def parse_driver_list_arg(drivers_arg):
    entries=[]
    if not drivers_arg: return entries
    for item in drivers_arg.split(','):
        item=item.strip()
        if not item: continue
        if ':' in item:
            d,c = item.split(':',1)
            entries.append({'driver': d.strip(), 'constructor': c.strip()})
        else:
            entries.append({'driver': item.strip(), 'constructor': None})
    return entries

def load_ml_table(path_ml):
    p = Path(path_ml)
    if not p.exists(): raise FileNotFoundError(f"ML table not found: {path_ml}")
    if p.suffix=='.parquet': return pd.read_parquet(p)
    return pd.read_csv(p)

def compute_rolling_stats(history_df, up_to_season, up_to_round):
    cond = ((history_df['season'] < up_to_season) |
            ((history_df['season'] == up_to_season) & (history_df['round'] < up_to_round)))
    df_hist = history_df[cond].copy()
    def rolling_for_driver(g):
        g = g.sort_values(['season','round'])
        g['avg_pos_last3'] = g['race_position_num'].rolling(3, min_periods=1).mean().shift(1)
        g['podiums_last3'] = g['podium'].rolling(3, min_periods=1).sum().shift(1)
        g['points_last3'] = g['points'].rolling(3, min_periods=1).sum().shift(1)
        return g
    df_roll = df_hist.groupby('driver', group_keys=False).apply(rolling_for_driver)
    latest = df_roll.sort_values(['driver','season','round']).groupby('driver').tail(1)
    stats = latest.set_index('driver')[['avg_pos_last3','podiums_last3','points_last3']].copy()
    return stats

def build_candidates_and_features(season, rnd, drivers_arg=None, qual_csv=None, ml_v1="data/ml_table.parquet", ml_v3="data/ml_table_v3.parquet"):
    # historical tables for medians & rolling stats
    hist_v1 = load_ml_table(ml_v1)
    hist_v3 = load_ml_table(ml_v3)
    hist_v1['season'] = pd.to_numeric(hist_v1['season'], errors='coerce')
    hist_v1['round']  = pd.to_numeric(hist_v1['round'], errors='coerce')
    hist_v3['season'] = pd.to_numeric(hist_v3['season'], errors='coerce')
    hist_v3['round']  = pd.to_numeric(hist_v3['round'], errors='coerce')

    candidates = parse_driver_list_arg(drivers_arg) if drivers_arg else []
    qtimes = {}

    # read qual_csv if provided
    if qual_csv:
        qf = Path(qual_csv)
        if qf.exists():
            qdf = pd.read_csv(qf, low_memory=False)
            qdf.columns = [c.lower() for c in qdf.columns]
            if 'abbreviation' in qdf.columns: qdf = qdf.rename(columns={'abbreviation':'driver'})
            if 'teamname' in qdf.columns: qdf = qdf.rename(columns={'teamname':'constructor'})
            for _, r in qdf.iterrows():
                drv = str(r.get('driver'))
                best = np.nan
                for col in ('best','q3','q2','q1'):
                    if col in qdf.columns:
                        v = time_to_seconds(r.get(col))
                        if not np.isnan(v) and (np.isnan(best) or v < best): best = v
                qtimes[drv] = best
                if not any(d['driver']==drv for d in candidates):
                    candidates.append({'driver':drv,'constructor': str(r.get('constructor')) if 'constructor' in r else None})

    # auto-fetch if no candidates
    if not candidates:
        fetched = fetch_session_csv(season, rnd, prefer='Q')
        if fetched:
            qdf = pd.read_csv(fetched, low_memory=False)
            qdf.columns = [c.lower() for c in qdf.columns]
            if 'abbreviation' in qdf.columns: qdf = qdf.rename(columns={'abbreviation':'driver'})
            if 'teamname' in qdf.columns: qdf = qdf.rename(columns={'teamname':'constructor'})
            for _, r in qdf.iterrows():
                drv = str(r.get('driver'))
                best = np.nan
                for col in ('best','q3','q2','q1'):
                    if col in qdf.columns:
                        v = time_to_seconds(r.get(col))
                        if not np.isnan(v) and (np.isnan(best) or v < best): best = v
                qtimes[drv] = best
                candidates.append({'driver': drv, 'constructor': str(r.get('constructor')) if 'constructor' in r else None})
        else:
            fetched_r = fetch_session_csv(season, rnd, prefer='R')
            if fetched_r:
                rdf = pd.read_csv(fetched_r, low_memory=False)
                rdf.columns = [c.lower() for c in rdf.columns]
                if 'abbreviation' in rdf.columns: rdf = rdf.rename(columns={'abbreviation':'driver'})
                if 'teamname' in rdf.columns: rdf = rdf.rename(columns={'teamname':'constructor'})
                for _, r in rdf.iterrows():
                    drv = str(r.get('driver'))
                    candidates.append({'driver': drv, 'constructor': str(r.get('constructor')) if 'constructor' in r else None})

    if not candidates:
        raise ValueError("No candidate drivers found. Provide --drivers or ensure FastF1 can fetch the session.")

    # compute rolling stats up to previous race (use v3/history for rows that match)
    rolling_stats = compute_rolling_stats(hist_v3, up_to_season=int(season), up_to_round=int(rnd))

    rows = []
    for c in candidates:
        drv = c['driver']
        cons = c.get('constructor')
        row = {'driver': drv, 'constructor': cons}
        if drv in rolling_stats.index:
            row.update(rolling_stats.loc[drv].to_dict())
        else:
            row['avg_pos_last3'] = hist_v3['race_position_num'].median()
            row['podiums_last3'] = 0.0
            row['points_last3'] = 0.0
        # q_time_s from qtimes or hist median
        if drv in qtimes and not np.isnan(qtimes[drv]):
            row['q_time_s'] = qtimes[drv]
        else:
            row['q_time_s'] = hist_v3['q_time_s'].median() if 'q_time_s' in hist_v3.columns else np.nan
        rows.append(row)

    feat_df = pd.DataFrame(rows)

    # compute q_delta_s (session normalized)
    if feat_df['q_time_s'].notna().any():
        smin = feat_df['q_time_s'].min(skipna=True)
        feat_df['q_delta_s'] = feat_df['q_time_s'] - smin
    else:
        feat_df['q_delta_s'] = np.nan

    # teammate delta for v3
    feat_df['constructor'] = feat_df['constructor'].astype(str)
    feat_df['team_min_qdelta'] = feat_df.groupby('constructor')['q_delta_s'].transform('min')
    feat_df['teammate_delta_q2'] = (feat_df['q_delta_s'] - feat_df['team_min_qdelta']).fillna(0.0)
    feat_df.drop(columns=['team_min_qdelta'], inplace=True)

    # compute q_position for v1: rank q_time_s (1 = fastest) if available
    if feat_df['q_time_s'].notna().any():
        feat_df['q_position'] = feat_df['q_time_s'].rank(method='min').astype(int)
    else:
        # fallback to historical median q_position if exists in hist_v1
        if 'q_position' in hist_v1.columns:
            feat_df['q_position'] = int(hist_v1['q_position'].median())
        else:
            feat_df['q_position'] = int(10)  # generic fallback

    # final imputation for rolling features
    for col in ['avg_pos_last3','podiums_last3','points_last3']:
        feat_df[col] = feat_df[col].fillna(hist_v3[col].median() if col in hist_v3.columns else 0.0)

    return feat_df

def run_ensemble_and_save(season, rnd, drivers_arg=None, qual_csv=None,
                          model1_path="models/xgb_podium_baseline_v1.joblib",
                          model3_path="models/xgb_podium_v3.joblib",
                          ml1="data/ml_table.parquet",
                          ml3="data/ml_table_v3.parquet"):
    # load models
    m1 = joblib.load(model1_path)
    m3 = joblib.load(model3_path)

    feat_df = build_candidates_and_features(season, rnd, drivers_arg, qual_csv, ml1, ml3)

    # prepare feature matrices
    FEATURES1 = ['q_position','avg_pos_last3','podiums_last3','points_last3']
    FEATURES3 = ['q_delta_s','teammate_delta_q2','avg_pos_last3','podiums_last3','points_last3']

    # ensure columns exist
    for f in FEATURES1:
        if f not in feat_df.columns: feat_df[f] = 0.0
    for f in FEATURES3:
        if f not in feat_df.columns: feat_df[f] = 0.0

    d1 = xgb.DMatrix(feat_df[FEATURES1].astype(float).fillna(0.0), feature_names=FEATURES1)
    d3 = xgb.DMatrix(feat_df[FEATURES3].astype(float).fillna(0.0), feature_names=FEATURES3)

    p1 = m1.predict(d1)
    p3 = m3.predict(d3)
    p_ens = (p1 + p3) / 2.0

    out = feat_df[['driver','constructor']].copy()
    out['prob_v1'] = p1
    out['prob_v3'] = p3
    out['prob_ens'] = p_ens
    out = out.sort_values('prob_ens', ascending=False).reset_index(drop=True)
    out['rank'] = out['prob_ens'].rank(method='dense', ascending=False).astype(int)

    # save to outputs/
    outdir = Path("outputs"); outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / f"predictions_season{season}_round{rnd}.csv"
    out.to_csv(out_file, index=False)
    print(f"\nEnsemble predictions saved -> {out_file}\n")
    print("Top 10:\n", out.head(10).to_string(index=False))

    return out, out_file

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--season", required=True, type=int)
    p.add_argument("--round", required=True, type=int)
    p.add_argument("--drivers", type=str, default=None, help='Comma "PIA:McLaren,VER:Red Bull"')
    p.add_argument("--qual_csv", type=str, default=None)
    p.add_argument("--model1", type=str, default="models/xgb_podium_baseline_v1.joblib")
    p.add_argument("--model3", type=str, default="models/xgb_podium_v3.joblib")
    p.add_argument("--ml1", type=str, default="data/ml_table.parquet")
    p.add_argument("--ml3", type=str, default="data/ml_table_v3.parquet")
    args = p.parse_args()

    run_ensemble_and_save(args.season, args.round, drivers_arg=args.drivers, qual_csv=args.qual_csv,
                          model1_path=args.model1, model3_path=args.model3, ml1=args.ml1, ml3=args.ml3)
