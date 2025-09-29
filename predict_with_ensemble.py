# predict_with_ensemble.py
import argparse
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path

def load_ml_table(path):
    if Path(path).suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def run_ensemble(season, rnd, model1_path, model2_path, ml_table1, ml_table2,
                 drivers=None, qual_csv=None):
    # Load models
    m1 = joblib.load(model1_path)
    m2 = joblib.load(model2_path)

    # Load ML tables (for imputation & medians)
    hist1 = load_ml_table(ml_table1)
    hist2 = load_ml_table(ml_table2)

    # Features v1
    FEATURES1 = ['q_position','avg_pos_last3','podiums_last3','points_last3']
    # Features v3
    FEATURES3 = ['q_delta_s','teammate_delta_q2','avg_pos_last3','podiums_last3','points_last3']

    # For now, just build candidates from latest season/round data
    # (you can extend this like predict_next_race.py later)
    test1 = hist1[(hist1['season']==season) & (hist1['round']==rnd)].copy()
    test3 = hist2[(hist2['season']==season) & (hist2['round']==rnd)].copy()

    # Align drivers
    drivers = sorted(set(test1['driver']).intersection(set(test3['driver'])))
    test1 = test1[test1['driver'].isin(drivers)]
    test3 = test3[test3['driver'].isin(drivers)]

    # Build matrices
    dtest1 = xgb.DMatrix(test1[FEATURES1].fillna(0), feature_names=FEATURES1)
    dtest3 = xgb.DMatrix(test3[FEATURES3].fillna(0), feature_names=FEATURES3)

    # Predict from both models
    p1 = m1.predict(dtest1)
    p3 = m2.predict(dtest3)

    # Simple average ensemble
    p_ens = (p1 + p3) / 2.0

    # Attach and sort
    out = pd.DataFrame({
        'driver': drivers,
        'prob_v1': p1,
        'prob_v3': p3,
        'prob_ens': p_ens
    }).sort_values('prob_ens', ascending=False)

    print(f"\nEnsemble predictions for {season} Round {rnd} (Top-10):\n")
    print(out.head(10).to_string(index=False, justify='left'))

    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--model1", default="models/xgb_podium_baseline_v1.joblib")
    p.add_argument("--model2", default="models/xgb_podium_v3.joblib")
    p.add_argument("--ml1", default="data/ml_table.parquet")
    p.add_argument("--ml2", default="data/ml_table_v3.parquet")
    args = p.parse_args()

    run_ensemble(args.season, args.round, args.model1, args.model2, args.ml1, args.ml2)
