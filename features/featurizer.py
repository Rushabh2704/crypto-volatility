import json
import argparse
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from datetime import datetime
import os

WINDOW_SIZE = 50

def parse_args():
    parser = argparse.ArgumentParser(description='Feature Engineering Consumer')
    parser.add_argument('--topic_in', type=str, default='ticks.raw')
    parser.add_argument('--topic_out', type=str, default='ticks.features')
    return parser.parse_args()

def extract_tick(message):
    """Pull the fields we need out of the raw Coinbase message"""
    try:
        events = message.get('events', [])
        if not events:
            return None
        ticker = events[0].get('tickers', [])
        if not ticker:
            return None
        t = ticker[0]
        return {
            'timestamp': message.get('timestamp'),
            'product_id': t.get('product_id'),
            'price': float(t.get('price', 0)),
            'best_bid': float(t.get('best_bid', 0)),
            'best_ask': float(t.get('best_ask', 0)),
            'best_bid_quantity': float(t.get('best_bid_quantity', 0)),
            'best_ask_quantity': float(t.get('best_ask_quantity', 0)),
            'volume_24_h': float(t.get('volume_24_h', 0)),
        }
    except Exception as e:
        print(f"Error extracting tick: {e}")
        return None

def compute_features(ticks: list) -> dict:
    """Compute features from a rolling window of ticks"""
    df = pd.DataFrame(ticks)

    # Midprice
    df['midprice'] = (df['best_bid'] + df['best_ask']) / 2

    # Midprice returns
    df['midprice_return'] = df['midprice'].pct_change()

    # Bid-ask spread
    df['spread'] = df['best_ask'] - df['best_bid']

    # Order book imbalance
    df['book_imbalance'] = (
        (df['best_bid_quantity'] - df['best_ask_quantity']) /
        (df['best_bid_quantity'] + df['best_ask_quantity'])
    )

    # Rolling volatility (std of returns over window)
    df['rolling_volatility'] = df['midprice_return'].rolling(WINDOW_SIZE, min_periods=2).std().fillna(0)

    # Trade intensity (volume change)
    df['volume_change'] = df['volume_24_h'].diff()

    # Get the latest row as our feature vector
    latest = df.iloc[-1].to_dict()
    latest['timestamp'] = ticks[-1]['timestamp']
    latest['product_id'] = ticks[-1]['product_id']

    return latest

def save_to_parquet(records: list):
    """Save feature records to parquet file"""
    os.makedirs('data/processed', exist_ok=True)
    df = pd.DataFrame(records)
    path = 'data/processed/features.parquet'
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} records to {path}")

def main():
    args = parse_args()

    consumer = KafkaConsumer(
        args.topic_in,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='latest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
    )

    print(f"Consuming from {args.topic_in}, publishing to {args.topic_out}")

    tick_buffer = []   # rolling window of raw ticks
    feature_records = []  # all computed features

    for message in consumer:
        tick = extract_tick(message.value)
        if tick is None:
            continue

        tick_buffer.append(tick)

        # Need at least WINDOW_SIZE ticks to compute rolling features
        if len(tick_buffer) < WINDOW_SIZE:
            print(f"Buffering ticks... {len(tick_buffer)}/{WINDOW_SIZE}")
            continue

        # Keep only the last WINDOW_SIZE ticks
        if len(tick_buffer) > WINDOW_SIZE:
            tick_buffer.pop(0)

        # Compute features
        features = compute_features(tick_buffer)
        feature_records.append(features)

        # Publish to Kafka
        producer.send(args.topic_out, value=features)
        vol = features.get('rolling_volatility')
        vol_str = f"{vol:.6f}" if vol == vol else "nan (price not moving)"
        print(f"Features computed: spread={features.get('spread'):.4f}, volatility={vol_str}")
        # Save to parquet every 100 records
        if len(feature_records) % 100 == 0:
            save_to_parquet(feature_records)

if __name__ == '__main__':
    main()