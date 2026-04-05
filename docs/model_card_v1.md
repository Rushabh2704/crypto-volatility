# Model Card v1 — Crypto Volatility Spike Detector

**Model:** xgboost_volatility
**Version:** 1.0
**Task:** Binary classification — detect 60-second BTC-USD volatility spikes

## 1. Model Details

Algorithm: XGBoost (gradient-boosted trees)
Artifact: models/artifacts/xgboost_model.json
MLflow experiment: crypto-volatility
Baseline comparator: Rolling volatility threshold rule, tuned on validation set

## 2. Intended Use

Primary use case: Early warning of short-term BTC-USD price volatility for alerting or downstream risk logic.

Out of scope:
- Automated trading or order execution
- Assets other than BTC-USD without retraining
- Prediction horizons other than 60 seconds

## 3. Training Data

Source: Coinbase Advanced Trade WebSocket API, public ticker feed
Asset: BTC-USD
Feature file: data/processed/features_labeled.parquet
Total rows: 13,039
Split: 70% train, 15% validation, 15% test, strictly chronological

## 4. Label Definition

volatility_proxy = rolling std of midprice returns over next 60 seconds
label = 1 if volatility_proxy >= tau else 0
tau = 0.000026, the 90th percentile of the training distribution
Label prevalence: approximately 10%

## 5. Features

midprice_return: mid-price return since last tick
spread: best_ask minus best_bid
book_imbalance: (bid_qty minus ask_qty) divided by (bid_qty plus ask_qty)
rolling_volatility: rolling std of midprice_return over a 50-tick window
volume_change: change in 24h volume since last tick

## 6. Evaluation Results

Primary metric is PR-AUC. Chosen because the positive class is rare at roughly 10%
and ROC-AUC is misleading on imbalanced data.

Volatility Threshold Baseline — Test PR-AUC: 0.1114, Test ROC-AUC: 0.5829
XGBoost — Test PR-AUC: 0.0893, Test ROC-AUC: 0.5077

The baseline outperformed XGBoost. This is common with small, noisy feature sets on
imbalanced financial time series. Both runs are logged in MLflow under crypto-volatility.

## 7. Drift Monitoring

Evidently report: reports/evidently/model_eval_drift_report.html
Compares training distribution (reference) vs test distribution (current).

book_imbalance: drift detected, Wasserstein score 0.195
spread: drift detected, Wasserstein score 0.169
volume_change: no drift, score 0.023
rolling_volatility: no drift, score 0.005
label: no drift, Jensen-Shannon score 0.044

book_imbalance and spread show meaningful distributional shift between train and test.
These should be monitored closely and trigger retraining if drift persists.

## 8. Limitations and Risks

Small feature set: only 5 features; adding order book depth would likely improve the model
Feature drift: book_imbalance and spread drift between windows
Single asset: trained only on BTC-USD, generalization to other pairs is unverified
Non-stationary market: crypto markets shift; regular retraining is recommended
Class imbalance: handled via scale_pos_weight in XGBoost and PR-AUC as primary metric
Inference speed: benchmarked at 0.000007x the 60-second real-time budget, well within limits

## 9. Ethical Considerations

Uses public market data only. No personal or private information is involved.
The model is not connected to any trading system.
All predictions require human review before any operational use.