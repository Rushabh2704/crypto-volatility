"""
infer.py — Score features using the saved XGBoost model.
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

FEATURE_COLS = [
    "midprice_return",
    "spread",
    "book_imbalance",
    "rolling_volatility",
    "volume_change",
]


def load_model():
    mp = ARTIFACTS / "xgboost_model.json"
    if not mp.exists():
        raise FileNotFoundError(f"Model not found at {mp}. Run train.py first.")
    model = xgb.XGBClassifier()
    model.load_model(str(mp))
    return model


def benchmark(model, n_rows=1000):
    X = np.random.randn(n_rows, len(FEATURE_COLS)).astype(np.float32)
    model.predict_proba(X)  # warmup
    t0 = time.perf_counter()
    model.predict_proba(X)
    ms = (time.perf_counter() - t0) * 1000
    ratio = (ms / 1000) / 60.0
    print(f"\n── Benchmark ({n_rows} rows) ──────────────────")
    print(f"  Total: {ms:.2f} ms  |  Per row: {ms/n_rows*1000:.2f} µs")
    print(f"  vs 60s budget: {ratio:.6f}x  {'✓ PASS' if ratio < 2.0 else '✗ FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",  default="data/processed/features_test.parquet")
    ap.add_argument("--output",    default="models/artifacts/predictions_output.parquet")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    model = load_model()
    print(f"Model loaded from {ARTIFACTS / 'xgboost_model.json'}")

    if args.benchmark:
        benchmark(model)

    fp = ROOT / args.features
    df = pd.read_parquet(fp).dropna(subset=FEATURE_COLS)
    print(f"Scoring {len(df)} rows ...")

    t0     = time.perf_counter()
    scores = model.predict_proba(df[FEATURE_COLS].values)[:, 1]
    ms     = (time.perf_counter() - t0) * 1000

    df["y_score"] = scores
    df["y_pred"]  = (scores >= args.threshold).astype(int)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    print(f"Done in {ms:.1f} ms — spike_rate={df['y_pred'].mean():.3f}")
    print(f"Predictions saved → {out}")

    if "label" in df.columns:
        from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
        p, r, _ = precision_recall_curve(df["label"], df["y_score"])
        print(f"PR-AUC={auc(r,p):.4f}  ROC-AUC={roc_auc_score(df['label'], df['y_score']):.4f}")


if __name__ == "__main__":
    main()