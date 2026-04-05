# Scoping Brief: Crypto Volatility Detection

## Use Case
Detect short-term volatility spikes in BTC-USD prices using real-time 
Coinbase market data streamed via Kafka. This system monitors live market 
conditions and flags when unusual price instability is likely in the 
next 60 seconds.

## 60-Second Prediction Goal
Predict whether BTC-USD price volatility (measured as rolling standard 
deviation of midprice returns) will exceed a threshold τ in the next 
60 seconds.

- Target horizon: 60 seconds
- Volatility proxy: rolling std of midprice returns
- Label: 1 if σ_future >= τ, else 0

## Success Metric
- Primary: PR-AUC (Precision-Recall Area Under Curve)
- Used because volatility spikes are rare — dataset will be imbalanced
- Target: PR-AUC > 0.5 on held-out test set
- Secondary: F1-score at chosen threshold

## Risk Assumptions
- Crypto trades 24/7 so no market hour gaps expected
- Spikes are rare events so accuracy alone is misleading
- Feature computation must complete faster than tick arrival rate
- Coinbase feed assumed reliable with no major outages
- System is for detection only — no trades are placed
- Model may degrade during extreme market events not seen in training