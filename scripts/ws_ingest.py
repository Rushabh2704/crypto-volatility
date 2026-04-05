import asyncio
import json
import argparse
import os
from datetime import datetime
from websockets.legacy.client import connect
from kafka import KafkaProducer

import ssl
import certifi

ssl_context = ssl.create_default_context(cafile=certifi.where())

COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"

def create_producer():
    return KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Coinbase WebSocket Ingestor')
    parser.add_argument('--pair', type=str, default='BTC-USD', help='Trading pair')
    parser.add_argument('--minutes', type=int, default=15, help='How long to run')
    return parser.parse_args()

async def ingest(pair: str, minutes: int, producer: KafkaProducer):
    os.makedirs('data/raw', exist_ok=True)
    filename = f"data/raw/{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ndjson"
    end_time = asyncio.get_event_loop().time() + (minutes * 60)

    subscribe_message = {
        "type": "subscribe",
        "product_ids": [pair],
        "channel": "ticker"
    }

    print(f"Connecting to Coinbase WebSocket...")
    async with connect(COINBASE_WS_URL, ssl=ssl_context) as websocket:
        await websocket.send(json.dumps(subscribe_message))
        print(f"Subscribed to {pair} ticker")

        with open(filename, 'w') as f:
            while asyncio.get_event_loop().time() < end_time:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10)
                    data = json.loads(message)

                    if data.get('channel') == 'ticker':
                        print(f"Tick received: {data}")
                        producer.send('ticks.raw', value=data)
                        f.write(json.dumps(data) + '\n')

                except asyncio.TimeoutError:
                    print("No message received, still waiting...")
                    continue

    producer.flush()
    print(f"Done! Data saved to {filename}")


def main():
    args = parse_args()
    producer = create_producer()
    print(f"Starting ingestion for {args.pair} for {args.minutes} minutes")

    while True:
        try:
            asyncio.run(ingest(args.pair, args.minutes, producer))
            print("Session complete. Restarting...")
        except Exception as e:
            print(f"Connection lost: {e}. Reconnecting in 5 seconds...")
            import time
            time.sleep(5)


if __name__ == '__main__':
    main()