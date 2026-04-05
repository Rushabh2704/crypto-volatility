import pandas as pd
import numpy as np

def add_labels(df: pd.DataFrame, horizon_seconds: int = 60, tau: float = None) -> pd.DataFrame:
    """
    For each row, compute volatility over the NEXT horizon_seconds worth of ticks.
    Label = 1 if sigma_future >= tau else 0
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    future_vols = []

    for i in range(len(df)):
        current_time = df.loc[i, 'timestamp']
        future_time = current_time + pd.Timedelta(seconds=horizon_seconds)

        # Get all ticks in the next 60 seconds
        future_mask = (df['timestamp'] > current_time) & (df['timestamp'] <= future_time)
        future_returns = df.loc[future_mask, 'midprice_return']

        if len(future_returns) >= 2:
            future_vols.append(future_returns.std())
        else:
            future_vols.append(np.nan)

    df['sigma_future'] = future_vols

    # Drop rows where we cant compute future volatility
    df = df.dropna(subset=['sigma_future'])

    # Set tau at 90th percentile if not provided
    if tau is None:
        tau = np.percentile(df['sigma_future'], 90)
        print(f"Threshold tau set at 90th percentile: {tau:.6f}")

    df['label'] = (df['sigma_future'] >= tau).astype(int)

    print(f"Total rows: {len(df)}")
    print(f"Spike rows (label=1): {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"Non-spike rows (label=0): {(df['label']==0).sum()}")

    return df, tau

if __name__ == '__main__':
    df = pd.read_parquet('data/processed/features.parquet')
    df_labeled, tau = add_labels(df)
    df_labeled.to_parquet('data/processed/features_labeled.parquet', index=False)
    print(f"\nSaved labeled data to data/processed/features_labeled.parquet")
    print(df_labeled['label'].value_counts())