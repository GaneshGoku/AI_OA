## Aave V2 Wallet Credit Scoring
🔷 Predicting credit scores (0–1000) for wallets interacting with Aave V2, based solely on historical transaction behavior.


## Overview

This project develops a robust credit scoring system for DeFi users on the Aave V2 protocol (Polygon), assigning each wallet a score between 0 and 1000.
Higher scores indicate reliable, responsible usage patterns; lower scores reflect risky, exploitative, or bot-like behavior.
It is based solely on historical transaction data, without external KYC or reputation signals.

## Methodology

## Problem framing:

We framed this as a regression problem, where the model predicts a numeric score per wallet based on engineered behavioral features.

## Model choice:

We chose XGBoost (eXtreme Gradient Boosting), a powerful tree-based ensemble method:

Robust to heterogeneous feature distributions

Captures non-linear interactions between features

Handles small and large datasets efficiently

Regularized to prevent overfitting

## Why not neural nets or linear models?

Neural nets are overkill given structured tabular data and synthetic labels

Linear models cannot capture non-linear risk patterns effectively

XGBoost is well-suited for credit scoring problems in practice.

                ┌──────────────────────────┐
                │  Raw JSON transactions   │
                │ user-wallet-transactions │
                └──────────────┬───────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │ Feature Engineering (Python script)│
            │  → Aggregate wallet-level features │
            └──────────────────┬──────────────────┘
                               │
               ┌───────────────▼───────────────┐
               │  Synthetic labels generation │
               │   (based on risk heuristics) │
               └───────────────┬──────────────┘
                               │
                ┌──────────────▼─────────────┐
                │   Model Training (XGBoost) │
                │   Input: features + labels │
                │   Output: trained model    │
                └──────────────┬─────────────┘
                               │
               ┌───────────────▼──────────────┐
               │ Scoring new wallets          │
               │   (with trained model)       │
               │ Output: wallet, credit_score│
               └──────────────────────────────┘


## Processing Flow

# Feature Engineering

Parse raw JSON file (data/user-wallet-transactions.json)

Aggregate transaction records into wallet-level behavioral features:

n_tx: total transactions

n_deposits, n_borrows, n_repays, n_liquidations

total_deposit_usd, total_borrow_usd, total_repay_usd

active_days: unique activity days

borrow_to_deposit_ratio

repay_to_borrow_ratio

liquidation_rate

Output: data/wallet_features.csv

# Synthetic Labeling & Model Training

Since true credit scores are not available, synthetic labels are generated based on domain-driven risk heuristics:

High liquidation rate → penalized

High borrow-to-deposit ratio (>0.7) → penalized

High repay-to-borrow ratio (~1.0) → rewarded

High activity & high volume → rewarded

Final score clipped to 0–1000, base = 700 ± 300.

Model: XGBoostRegressor

Trained on engineered features + synthetic labels.

Model saved to: models/credit_score_xgb.pkl

# Scoring New Wallets

Use trained model to predict credit scores for each wallet.

Save results to: data/wallet_scores.csv

## How to Run

pip install -r requirements.txt

python src/feature_engineering.py

python src/train_xgb.py

python src/predict_scores.py


