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
