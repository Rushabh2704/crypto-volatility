import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import os

os.makedirs('reports/evidently', exist_ok=True)

# Load labeled data
df = pd.read_parquet('data/processed/features_labeled.parquet', engine='pyarrow')
df = df.sort_values('timestamp').reset_index(drop=True)

# Select numeric features only
feature_cols = ['midprice', 'midprice_return', 'spread',
                'book_imbalance', 'rolling_volatility', 'volume_change']

df_features = df[feature_cols].fillna(0)

# Split into early and late windows
split = len(df_features) // 2
reference = df_features.iloc[:split]
current = df_features.iloc[split:]

print(f"Reference window: {len(reference)} rows")
print(f"Current window: {len(current)} rows")

# Generate report
report = Report([
    DataDriftPreset(),
])

my_report = report.run(reference_data=reference, current_data=current)
my_report.save_html('reports/evidently/data_report.html')
print("Report saved to reports/evidently/data_report.html")