import pandas as pd
import xgboost as xgb
import joblib
from feature_engineering import load_and_engineer
import sklearn
import numpy as np
# load features
df = load_and_engineer("C:/Users/Sonwa/AI_OA/data/user-wallet-transactions.json")

# create synthetic label
df["credit_score"] = (
    700
    - (df["liquidation_rate"].clip(upper=1.0) * 400)
    - ((df["borrow_to_deposit_ratio"] - 0.7).clip(lower=0) * 200)
    + (df["repay_to_borrow_ratio"].clip(upper=1.0) * 200)
    + (df["activity_days"].clip(upper=30) / 30 * 50)
    + (np.log1p(df["total_deposit_usd"] + df["total_borrow_usd"]) / 10).clip(upper=50)
).clip(0, 1000)


df.to_csv("C:/Users/Sonwa/AI_OA/data/wallet_features.csv", index=False)

X = df.drop(columns=["wallet", "credit_score"])
y = df["credit_score"]

model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "C:/Users/Sonwa/AI_OA/model/credit_score_xgb.pkl")
print(" Model trained and saved to models/credit_score_xgb.pkl")
