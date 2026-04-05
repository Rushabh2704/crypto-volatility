"""
evidently_model_report.py — Evidently report comparing train vs test distributions.
"""

import pandas as pd
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "midprice_return",
    "spread",
    "book_imbalance",
    "rolling_volatility",
    "volume_change",
]
LABEL_COL = "label"

def main():
    fp = ROOT / "data/processed/features_labeled.parquet"
    df = pd.read_parquet(fp).dropna(subset=FEATURE_COLS + [LABEL_COL])
    df = df.sort_values("timestamp")

    n = len(df)
    train_df = df.iloc[:int(n * 0.70)]
    test_df  = df.iloc[int(n * 0.85):]

    cols = FEATURE_COLS + [LABEL_COL]
    ref  = train_df[cols].reset_index(drop=True)
    cur  = test_df[cols].reset_index(drop=True)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=ref, current_data=cur)

    out_dir = ROOT / "reports/evidently"
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / "model_eval_drift_report.html"
    result.save_html(str(html_path))
    print(f"Report saved → {html_path}")

if __name__ == "__main__":
    main()