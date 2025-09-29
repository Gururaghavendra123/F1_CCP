# train_with_quali_teammate.py
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

# load
if os.path.exists("data/ml_table_v2.parquet"):
    df = pd.read_parquet("data/ml_table_v2.parquet")
else:
    df = pd.read_csv("data/ml_table_v2.csv")

print("Loaded:", df.shape)
df['season'] = pd.to_numeric(df['season'], errors='coerce')

# features (include q_time_s and teammate delta)
FEATURES = ['q_time_s','teammate_delta_q','avg_pos_last3','podiums_last3','points_last3']

# prepare dataset
for f in FEATURES:
    df[f] = pd.to_numeric(df.get(f, 0), errors='coerce').fillna(df[f].median() if df[f].notna().any() else 0)

train = df[df['season'] <= 2023].reset_index(drop=True)
test  = df[df['season'] >= 2024].reset_index(drop=True)
print("Train/Test:", train.shape, test.shape)

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
model = xgb.train(params, dtrain, num_boost_round=400, verbose_eval=50)
preds = model.predict(dtest)

# metrics
ll = log_loss(test['podium'], preds)
print("Logloss:", ll)

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
print("Top-3 accuracy:", t3)

# save
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_podium_v2.joblib")
print("Saved model -> models/xgb_podium_v2.joblib")

# feature importance
fi = model.get_score(importance_type='gain')
fi_items = sorted(fi.items(), key=lambda x: x[1], reverse=True)
if fi_items:
    names = [x[0] for x in fi_items]; vals = [x[1] for x in fi_items]
    plt.figure(figsize=(6,3))
    plt.barh(names[::-1], vals[::-1])
    plt.tight_layout()
    plt.savefig("models/feature_importance_v2.png", dpi=150)
    print("Feature importance saved -> models/feature_importance_v2.png")
else:
    print("No feature importance available.")
