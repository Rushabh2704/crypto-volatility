# Feature Specification: Crypto Volatility Detection

## Target
- Target horizon: 60 seconds
- Volatility proxy: rolling standard deviation of midprice returns 
  over the next 60 seconds
- Label definition: 1 if σ_future >= τ; else 0
- Chosen threshold τ: 0.000027 (90th percentile of σ_future distribution)

## Input Data
- Source: Coinbase Advanced Trade WebSocket API
- Trading pair: BTC-USD
- Channel: ticker
- Storage: NDJSON (raw), Parquet (processed)

## Features

### 1. midprice
- Formula: (best_bid + best_ask) / 2
- Description: Middle point between best bid and ask price
- Type: float

### 2. midprice_return
- Formula: (midprice_t - midprice_t-1) / midprice_t-1
- Description: Percentage change in midprice between ticks
- Type: float

### 3. spread
- Formula: best_ask - best_bid
- Description: Difference between best ask and best bid
- Type: float

### 4. book_imbalance
- Formula: (best_bid_quantity - best_ask_quantity) / 
           (best_bid_quantity + best_ask_quantity)
- Description: Measures buying vs selling pressure at top of book
- Range: -1 (all ask) to +1 (all bid)
- Type: float

### 5. rolling_volatility
- Formula: std(midprice_return) over last 10 ticks
- Description: Short term price instability measure
- Type: float

### 6. volume_change
- Formula: volume_24h_t - volume_24h_t-1
- Description: Change in 24h volume between ticks
- Type: float

## Rolling Window
- Window size: 10 ticks
- Approximately 2-3 seconds of data per window

## Label Construction
- For each row, look forward 60 seconds of ticks
- Compute σ_future = std(midprice_return) over that forward window
- Label = 1 if σ_future >= τ else 0
- τ will be set at the 90th percentile of σ_future distribution
  (to be confirmed in EDA notebook)

## Train/Validation/Test Split
- Split method: time-based (no shuffling)
- Train: first 70% of data
- Validation: next 15%
- Test: final 15%