# Crypto Volatility Spike Detection Pipeline

Real-time BTC-USD volatility detection using Kafka, MLflow, XGBoost, and Evidently.

## Quick Start
```bash
# 0. Environment
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Start infrastructure
docker compose -f docker/compose.yaml up -d

# 2. Ingest 15 minutes of ticks
python scripts/ws_ingest.py --pair BTC-USD --minutes 15

# 3. Validate stream
python scripts/kafka_consume_check.py --topic ticks.raw --min 100

# 4. Build features
python features/featurizer.py --topic_in ticks.raw --topic_out ticks.features

# 5. Add labels
python features/add_labels.py

# 6. Train models
python models/train.py --mlflow-uri ./mlruns

# 7. Score test set
python models/infer.py --features data/processed/features_test.parquet --benchmark

# 8. Generate Evidently drift report
python scripts/evidently_model_report.py

# 9. View MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5001
```

## Repository Layout
/data/raw/               Raw tick data (NDJSON)
/data/processed/         Feature parquet files
/features/               featurizer.py, add_labels.py
/models/                 train.py, infer.py, artifacts/
/notebooks/              eda.ipynb
/reports/                model_eval.pdf, evidently/ reports
/scripts/                ws_ingest.py, replay.py, kafka_consume_check.py
/docker/                 compose.yaml, Dockerfile.ingestor
/docs/                   scoping_brief, feature_spec, model_card, genai_appendix
/handoff/                Team handoff files
mlruns/                  MLflow experiment store

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Primary metric | PR-AUC | ~10% label prevalence; ROC-AUC misleading on imbalanced data |
| Label threshold tau | 0.000026 | 90th percentile of training volatility proxy |
| Train/val/test split | 70/15/15 | Strictly chronological; prevents look-ahead bias |
| Class imbalance | scale_pos_weight | Built-in XGBoost mechanism |
| Baseline | Rolling volatility threshold | Simple, interpretable, tunable |

## Results

| Model | Test PR-AUC | Test ROC-AUC |
|---|---|---|
| Volatility Threshold Baseline | 0.1114 | 0.5829 |
| XGBoost | 0.0893 | 0.5077 |

## Milestone Status

- [x] M1: Kafka + MLflow, WebSocket ingestor, Kafka consumer, scoping brief
- [x] M2: Featurizer, EDA notebook, Evidently report, feature spec, label definition
- [x] M3: train.py, infer.py, MLflow logging, model card, drift report, model_eval.pdf, handoff