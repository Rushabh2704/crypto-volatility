import json
import argparse
from kafka import KafkaConsumer

def parse_args():
    parser = argparse.ArgumentParser(description='Kafka Consumer Check')
    parser.add_argument('--topic', type=str, default='ticks.raw')
    parser.add_argument('--min', type=int, default=100, help='Minimum messages to confirm')
    return parser.parse_args()

def main():
    args = parse_args()
    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )

    print(f"Listening to topic: {args.topic}")
    count = 0
    for message in consumer:
        count += 1
        print(f"Message {count}: {message.value.get('timestamp', 'no timestamp')}")
        if count >= args.min:
            print(f"\n Success! Received {count} messages from {args.topic}")
            break

if __name__ == '__main__':
    main()