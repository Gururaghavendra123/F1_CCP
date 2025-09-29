# retrain_v3_and_eval.py
import pandas as pd, xgboost as xgb, joblib, os
from sklearn.metrics import log_loss
import numpy as np

# load ml_table_v3
df = pd.read_parquet("data/ml_table_v3.parquet")
df['season'] = pd.to_numeric(df['season'], errors='coerce')

# features: session-normalized q_delta and teammate delta
FEATURES = ['q_delta_s','teammate_delta_q2','avg_pos_last3','podiums_last3','points_last3']

# ensure numeric and reasonable imputation
for f in FEATURES:
    if f not in df.columns:
        df[f] = 0.0
    df[f] = pd.to_numeric(df[f], errors='coerce')
    df[f] = df[f].fillna(df[f].median())

# split: train seasons <= 2023, test >= 2024
train = df[df['season'] <= 2023].reset_index(drop=True)
test  = df[df['season'] >= 2024].reset_index(drop=True)

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

print("Training model...")
model = xgb.train(params, dtrain, num_boost_round=400, verbose_eval=50)

# predictions & metrics
preds_train = model.predict(dtrain)
preds_test  = model.predict(dtest)
ll_train = log_loss(train['podium'], preds_train)
ll_test  = log_loss(test['podium'], preds_test)

def top3_accuracy(df_sub, preds):
    df2 = df_sub.copy(); df2['pred'] = preds
    accs=[]
    for (s,r), g in df2.groupby(['season','round']):
        top3 = g.sort_values('pred', ascending=False).head(3)
        actual = set(g[g['podium']==1]['driver'])
        accs.append(len(actual & set(top3['driver']))/3.0)
    return np.mean(accs) if accs else np.nan

t3_train = top3_accuracy(train, preds_train)
t3_test  = top3_accuracy(test, preds_test)

print(f"Train Logloss: {ll_train:.4f}  |  Test Logloss: {ll_test:.4f}")
print(f"Train Top-3: {t3_train:.4f}       |  Test Top-3: {t3_test:.4f}")

# save model and feature importance
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_podium_v3.joblib")
print("Saved model -> models/xgb_podium_v3.joblib")

fi = model.get_score(importance_type='gain')
print("Feature importance (gain):", sorted(fi.items(), key=lambda x: x[1], reverse=True))
