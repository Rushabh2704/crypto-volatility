# Handoff Note

Contributor: Rushabh Kankariya
Integration option: Neither

## What is included

compose.yaml and Dockerfile.ingestor: Docker setup for Kafka and MLflow
.env.example: environment variable template
feature_spec.md: feature definitions and label definition
model_card_v1.md: full model card for the XGBoost classifier
requirements.txt: all Python dependencies
BTC-USD_20260404_171343.ndjson: 10-minute raw tick slice
test_predictions.parquet: model predictions on the held-out test set
model_eval.pdf: full evaluation report
model_eval_drift_report.html: Evidently train vs test drift report

## Quick start

cp .env.example .env
docker compose up -d
python models/infer.py --features data/processed/features_test.parquet --benchmark

## Key decisions

Label threshold tau = 0.000026, the 90th percentile of training volatility
Primary metric: PR-AUC, chosen due to 10% label prevalence
Splits: 70/15/15 strictly chronological, no shuffling
Baseline beats XGBoost on this dataset due to small feature set
book_imbalance and spread show drift between train and test windows