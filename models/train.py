"""
train.py — Train baseline and XGBoost models for volatility spike detection.
Logs parameters, metrics, and artifacts to MLflow. Uses time-based splits.
"""

import argparse
import json
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    f1_score,
    classification_report,
    roc_auc_score,
)
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT          = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "midprice_return",
    "spread",
    "book_imbalance",
    "rolling_volatility",
    "volume_change",
]
LABEL_COL = "label"


def load_and_split(parquet_path):
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    df = df.sort_values("timestamp")

    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    print(f"Split sizes  — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print(f"Label rates  — train: {train[LABEL_COL].mean():.3f}, "
          f"val: {val[LABEL_COL].mean():.3f}, test: {test[LABEL_COL].mean():.3f}")
    return train, val, test


def xy(df):
    return df[FEATURE_COLS].values, df[LABEL_COL].values


def compute_metrics(y_true, y_score, threshold=0.5):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    roc    = roc_auc_score(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    return {"pr_auc": pr_auc, "roc_auc": roc, f"f1_at_{threshold}": f1}


# ── Baseline: rolling volatility threshold ─────────────────────────────────────
class VolatilityBaseline:
    """Predict spike=1 when rolling_volatility exceeds threshold (soft sigmoid score)."""

    def __init__(self, thresh=0.000026):
        self.thresh = thresh

    def score(self, df):
        v = df["rolling_volatility"].values
        # scale factor so sigmoid is sensitive at these tiny values
        return 1 / (1 + np.exp(-(v - self.thresh) * 100000))

    def get_params(self):
        return {"thresh": self.thresh}


def train_baseline(train_df, val_df, test_df, mlflow_uri, experiment):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)

    # tune threshold on validation set
    candidates = [0.000010, 0.000020, 0.000026, 0.000035, 0.000050]
    best, best_thresh = -1, 0.000026
    for t in candidates:
        m = compute_metrics(val_df[LABEL_COL].values, VolatilityBaseline(t).score(val_df))
        print(f"  thresh={t:.6f}  val_pr_auc={m['pr_auc']:.4f}")
        if m["pr_auc"] > best:
            best, best_thresh = m["pr_auc"], t

    model = VolatilityBaseline(best_thresh)
    print(f"  → Best threshold: {best_thresh}")

    with mlflow.start_run(run_name="baseline_volatility_threshold"):
        mlflow.log_params({"model_type": "volatility_baseline", **model.get_params()})

        for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            m = compute_metrics(df[LABEL_COL].values, model.score(df))
            mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items()})
            if split == "test":
                test_m = m

        p = ARTIFACTS_DIR / "baseline_params.json"
        p.write_text(json.dumps(model.get_params(), indent=2))
        mlflow.log_artifact(str(p), artifact_path="baseline")
        run_id = mlflow.active_run().info.run_id

    print(f"\n[Baseline]  test PR-AUC={test_m['pr_auc']:.4f}  ROC-AUC={test_m['roc_auc']:.4f}")
    return model, test_m, run_id


# ── ML model: XGBoost ──────────────────────────────────────────────────────────
def train_xgboost(train_df, val_df, test_df, mlflow_uri, experiment):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)

    X_tr, y_tr = xy(train_df)
    X_va, y_va = xy(val_df)
    X_te, y_te = xy(test_df)

    spw = round((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 2)

    params = {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": spw, "random_state": 42, "n_jobs": -1,
    }

    model = xgb.XGBClassifier(**params, eval_metric="aucpr")
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    with mlflow.start_run(run_name="xgboost_volatility"):
        mlflow.log_params({"model_type": "xgboost", "features": ",".join(FEATURE_COLS), **params})

        for split, X, y in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]:
            scores = model.predict_proba(X)[:, 1]
            m = compute_metrics(y, scores)
            mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items()})
            if split == "test":
                test_m, test_scores = m, scores

        # Feature importance
        fi = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
        fi_path = ARTIFACTS_DIR / "feature_importance.json"
        fi_path.write_text(json.dumps(fi, indent=2))
        mlflow.log_artifact(str(fi_path), artifact_path="xgboost")

        # Save model
        mp = ARTIFACTS_DIR / "xgboost_model.json"
        model.save_model(str(mp))
        mlflow.log_artifact(str(mp), artifact_path="xgboost")
        mlflow.xgboost.log_model(model, artifact_path="xgboost_mlflow_model")

        # Save test predictions
        preds = test_df[FEATURE_COLS].copy()
        preds["y_true"]  = y_te
        preds["y_score"] = test_scores
        preds["y_pred"]  = (test_scores >= 0.5).astype(int)
        pp = ARTIFACTS_DIR / "test_predictions.parquet"
        preds.to_parquet(pp)
        mlflow.log_artifact(str(pp), artifact_path="xgboost")

        run_id = mlflow.active_run().info.run_id

    print(f"\n[XGBoost]   test PR-AUC={test_m['pr_auc']:.4f}  ROC-AUC={test_m['roc_auc']:.4f}")
    print(classification_report(y_te, (test_scores >= 0.5).astype(int), zero_division=0))
    return model, test_m, run_id


def save_test_split(test_df):
    info = {"feature_cols": FEATURE_COLS, "label_col": LABEL_COL, "test_rows": len(test_df)}
    (ARTIFACTS_DIR / "split_info.json").write_text(json.dumps(info, indent=2))
    out = ROOT / "data" / "processed" / "features_test.parquet"
    test_df.to_parquet(out)
    print(f"Test set saved → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",   default="data/processed/features_labeled.parquet")
    ap.add_argument("--mlflow-uri", default="http://localhost:5001")
    ap.add_argument("--experiment", default="crypto-volatility")
    args = ap.parse_args()

    fp = ROOT / args.features
    print(f"Loading: {fp}")

    train_df, val_df, test_df = load_and_split(str(fp))
    save_test_split(test_df)

    print("\n── Baseline ─────────────────────────────────────────────")
    bm, bm_metrics, bm_run = train_baseline(train_df, val_df, test_df, args.mlflow_uri, args.experiment)

    print("\n── XGBoost ──────────────────────────────────────────────")
    xm, xm_metrics, xm_run = train_xgboost(train_df, val_df, test_df, args.mlflow_uri, args.experiment)

    print("\n── Summary ──────────────────────────────────────────────")
    print(f"{'Model':<25} {'PR-AUC':>8} {'ROC-AUC':>9}")
    print(f"{'Baseline (vol threshold)':<25} {bm_metrics['pr_auc']:>8.4f} {bm_metrics['roc_auc']:>9.4f}")
    print(f"{'XGBoost':<25} {xm_metrics['pr_auc']:>8.4f} {xm_metrics['roc_auc']:>9.4f}")
    print(f"\nView runs at {args.mlflow_uri}")


if __name__ == "__main__":
    main()