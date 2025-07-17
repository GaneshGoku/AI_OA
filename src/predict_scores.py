import pandas as pd
import joblib
from feature_engineering import load_and_engineer
import numpy as np

# load features
df = load_and_engineer("C:/Users/Sonwa/AI_OA/data/user-wallet-transactions.json")

# load trained model
model = joblib.load("C:/Users/Sonwa/AI_OA/model/credit_score_xgb.pkl")

X = df.drop(columns=["wallet"])

preds = model.predict(X)
preds = np.clip(preds, 0, 1000)

df_scores = pd.DataFrame({
    "wallet": df["wallet"],
    "credit_score": preds.astype(int)
})

df_scores.to_csv("C:/Users/Sonwa/AI_OA/data/wallet_score.csv", index=False)
print("✅ Predicted scores saved to data/wallet_score.csv")
