# Model Evaluation Report
## Crypto Volatility Spike Detection — BTC-USD
### Milestone 3

---

## 1. Overview

This report evaluates two models trained to detect short-term volatility spikes in
BTC-USD tick data streamed from the Coinbase Advanced Trade WebSocket API.

Task: binary classification — predict whether rolling std of mid-price returns over
the next 60 seconds will exceed tau = 0.000026 (90th percentile of training data).

---

## 2. Dataset

Asset: BTC-USD
Source: Coinbase Advanced Trade WebSocket, public ticker
Total rows after cleaning: 13,039
Features: 5 engineered features
Label prevalence: approximately 10%
Train: 70% (9,127 rows)
Validation: 15% (1,956 rows)
Test: 15% (1,956 rows)
All splits are strictly chronological to prevent look-ahead bias.

---

## 3. Models

Baseline: Rolling volatility threshold rule. Predicts spike=1 when rolling_volatility
exceeds a tuned threshold. Threshold selected by grid search over validation PR-AUC.
Best threshold: 0.000035.

XGBoost: Gradient-boosted classifier trained on all 5 features.
Key parameters: n_estimators=300, max_depth=4, learning_rate=0.05,
scale_pos_weight=8.0 to handle class imbalance.

---

## 4. Results

Primary metric: PR-AUC. Chosen because the positive class is rare at ~10% and
ROC-AUC is misleading on imbalanced datasets.

Baseline (volatility threshold): Test PR-AUC = 0.1114, Test ROC-AUC = 0.5829
XGBoost: Test PR-AUC = 0.0893, Test ROC-AUC = 0.5077

The baseline outperformed XGBoost. This is a known phenomenon with small feature
sets on noisy financial time series. Both runs are logged in MLflow.

---

## 5. Inference Speed

XGBoost inference on 1,000 rows: 0.43ms total, 0.43 microseconds per row.
Ratio vs 60-second real-time budget: 0.000007x. Requirement of less than 2x: PASS.

---

## 6. Drift Analysis

Evidently report compares training vs test feature distributions.
2 out of 6 columns showed drift (33%).

book_imbalance: drift detected, Wasserstein score 0.195
spread: drift detected, Wasserstein score 0.169
volume_change: no drift, score 0.023
rolling_volatility: no drift, score 0.005
label: no drift, Jensen-Shannon score 0.044

book_imbalance and spread shift meaningfully between the training and test windows.
These features should trigger retraining if drift persists in production.

---

## 7. Limitations

Only 5 features available — adding order book depth would strengthen the signal.
Trained on a single short collection window — longer ingestion would improve robustness.
book_imbalance and spread drift — monitor and retrain regularly.
Model trained on BTC-USD only — retrain before applying to other assets.

---

## 8. Reproducibility

docker compose up -d
python scripts/ws_ingest.py --pair BTC-USD --minutes 15
python features/featurizer.py
python features/add_labels.py
python models/train.py --mlflow-uri ./mlruns
python models/infer.py --features data/processed/features_test.parquet --benchmark
python scripts/evidently_model_report.py