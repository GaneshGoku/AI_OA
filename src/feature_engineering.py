import json
import pandas as pd
import numpy as np

def load_and_engineer(json_file):
    with open(json_file) as f:
        raw = json.load(f)

    records = []
    for tx in raw:
        wallet = tx["userWallet"]
        timestamp = pd.to_datetime(tx["timestamp"], unit="s")
        action = tx["action"]
        amount_raw = float(tx["actionData"]["amount"])
        price_usd = float(tx["actionData"].get("assetPriceUSD", 1.0))
        symbol = tx["actionData"].get("assetSymbol", "")
        decimals = 6 if symbol in ["USDC", "USDT"] else 18

        amount_usd = (amount_raw / (10**decimals)) * price_usd

        records.append({
            "wallet": wallet,
            "timestamp": timestamp,
            "action": action,
            "amount_usd": amount_usd,
            "day": timestamp.date()
        })

    df = pd.DataFrame(records)

    grouped = df.groupby("wallet")

    features = grouped.agg(
        n_transactions=("action", "count"),
        n_deposits=("action", lambda x: (x == "deposit").sum()),
        n_borrows=("action", lambda x: (x == "borrow").sum()),
        n_repays=("action", lambda x: (x == "repay").sum()),
        n_liquidations=("action", lambda x: (x == "liquidationcall").sum()),
        total_deposit_usd=("amount_usd", lambda x: x[df.loc[x.index, "action"] == "deposit"].sum()),
        total_borrow_usd=("amount_usd", lambda x: x[df.loc[x.index, "action"] == "borrow"].sum()),
        total_repay_usd=("amount_usd", lambda x: x[df.loc[x.index, "action"] == "repay"].sum()),
        activity_days=("day", "nunique")
    ).reset_index()

    features["borrow_to_deposit_ratio"] = (
        features["total_borrow_usd"] / features["total_deposit_usd"].replace(0, np.nan)
    )
    features["repay_to_borrow_ratio"] = (
        features["total_repay_usd"] / features["total_borrow_usd"].replace(0, np.nan)
    )
    features["liquidation_rate"] = (
        features["n_liquidations"] / features["n_borrows"].replace(0, np.nan)
    )

    features.fillna(0, inplace=True)

    return features

if __name__ == "__main__":
    df = load_and_engineer("C:/Users/Sonwa/AI_OA/data/user-wallet-transactions.json")
    df.to_csv("C:/Users/Sonwa/AI_OA/data/wallet_features.csv", index=False)
    print("✅ Wallet features saved to data/wallet_features.csv")
