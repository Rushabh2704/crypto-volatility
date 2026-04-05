import json
import glob
import argparse
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.featurizer import extract_tick, compute_features, WINDOW_SIZE

def parse_args():
    parser = argparse.ArgumentParser(description='Replay raw ticks and regenerate features')
    parser.add_argument('--raw', type=str, default='data/raw/*.ndjson')
    parser.add_argument('--out', type=str, default='data/processed/features.parquet')
    return parser.parse_args()

def main():
    args = parse_args()

    # Find all raw files
    files = glob.glob(args.raw)
    if not files:
        print(f"No files found at {args.raw}")
        return

    print(f"Found {len(files)} raw file(s)")

    tick_buffer = []
    feature_records = []

    for filepath in sorted(files):
        print(f"Replaying {filepath}")
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                message = json.loads(line)
                tick = extract_tick(message)
                if tick is None:
                    continue

                tick_buffer.append(tick)

                if len(tick_buffer) < WINDOW_SIZE:
                    continue

                if len(tick_buffer) > WINDOW_SIZE:
                    tick_buffer.pop(0)

                features = compute_features(tick_buffer)
                feature_records.append(features)

    # Save to parquet
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(feature_records)
    df.to_parquet(args.out, index=False)
    print(f"Saved {len(df)} feature records to {args.out}")

if __name__ == '__main__':
    main()