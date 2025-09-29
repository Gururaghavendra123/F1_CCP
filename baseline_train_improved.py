# baseline_train_improved.py
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

# --- load ML table (parquet preferred) ---
if os.path.exists("data/ml_table.parquet"):
    df = pd.read_parquet("data/ml_table.parquet")
elif os.path.exists("data/ml_table.csv"):
    df = pd.read_csv("data/ml_table.csv")
else:
    raise FileNotFoundError("data/ml_table.parquet or data/ml_table.csv not found")

print("Loaded ML table:", df.shape)
print("Seasons present:", sorted(df['season'].dropna().unique()))

# --- features & basic cleaning ---
FEATURES = ['q_position','avg_pos_last3','podiums_last3','points_last3']
for f in FEATURES:
    df[f] = pd.to_numeric(df.get(f, 0), errors='coerce').fillna(-1)

# split by time: train seasons <= 2023, test >= 2024 (adjust as you want)
train = df[df['season'] <= 2023].reset_index(drop=True)
test  = df[df['season'] >= 2024].reset_index(drop=True)
print("Train / Test shapes:", train.shape, test.shape)

dtrain = xgb.DMatrix(train[FEATURES], label=train['podium'])
dtest  = xgb.DMatrix(test[FEATURES], label=test['podium'])

params = {
    'objective':'binary:logistic',
    'eval_metric':'logloss',
    'tree_method':'hist',
    'learning_rate':0.05,
    'max_depth':6,
    'subsample':0.8,
    'colsample_bytree':0.7,
    'seed':42
}

model = xgb.train(params, dtrain, num_boost_round=300, verbose_eval=50)
preds = model.predict(dtest)

# Logloss
ll = log_loss(test['podium'], preds)
print(f"Logloss on test: {ll:.4f}")

# Top-3 accuracy
def top3_accuracy(df_test, preds):
    df2 = df_test.copy()
    df2['pred'] = preds
    accs=[]
    for (s,r), g in df2.groupby(['season','round']):
        top3 = g.sort_values('pred', ascending=False).head(3)
        actual = set(g[g['podium']==1]['driver'])
        predicted = set(top3['driver'])
        accs.append(len(actual & predicted)/3.0)
    return np.mean(accs)

t3 = top3_accuracy(test, preds)
print(f"Top-3 accuracy (test): {t3:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_podium_baseline_v1.joblib")
print("Saved model -> models/xgb_podium_baseline_v1.joblib")

# Feature importance (plot)
fi = model.get_score(importance_type='gain')
fi_items = sorted(fi.items(), key=lambda x: x[1], reverse=True)
names = [x[0] for x in fi_items]
vals  = [x[1] for x in fi_items]

if names:
    plt.figure(figsize=(6,3))
    plt.barh(names[::-1], vals[::-1])
    plt.title("XGBoost feature importance (gain)")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png", dpi=150)
    print("Feature importance saved -> models/feature_importance.png")
else:
    print("No feature importance available (empty).")
