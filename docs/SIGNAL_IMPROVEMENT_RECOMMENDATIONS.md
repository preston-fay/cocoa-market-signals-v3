# Signal Detection Performance Improvement Recommendations

## Current Status
- Signal Accuracy: 53.5% (barely better than random)
- Sharpe Ratio: 0.69 (suboptimal)
- Models tested: 7 (all showing mediocre performance)

## Root Cause Analysis

### 1. Feature Engineering Issues
- **Current features are too basic**: Simple anomalies and changes
- **Missing interaction features**: Weather × Trade interactions
- **No lagged features**: Previous month's signals not used
- **No engineered ratios**: Price/Export ratios, momentum indicators

### 2. Model Selection Problems
- **Using generic models**: Not tailored for time series
- **No ensemble methods**: Individual models instead of combined
- **Missing specialized models**: LSTM, Prophet, or specialized commodity models

### 3. Data Quality & Quantity
- **Limited historical data**: Only 2 years
- **Monthly granularity for some features**: Daily would be better
- **Missing key data sources**:
  - Futures market positioning (COT reports)
  - Options flow data
  - Shipping/freight rates
  - Currency movements (USD strength)
  - Inventory/stock levels

## Recommended Improvements

### 1. Enhanced Feature Engineering
```python
# Example: Create interaction features
df['weather_trade_interaction'] = df['rainfall_anomaly'] * df['trade_volume_change']
df['price_momentum_30d'] = df['price'].pct_change(30)
df['price_acceleration'] = df['price_momentum_30d'].diff()
df['export_concentration_change'] = df['export_concentration'].diff()

# Seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['price'], model='multiplicative', period=30)
df['price_trend'] = decomposition.trend
df['price_seasonal'] = decomposition.seasonal
```

### 2. Advanced Models
```python
# 1. LSTM for time series
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 2. Prophet for forecasting
from prophet import Prophet

# 3. XGBoost with time-aware features
import xgboost as xgb
# Use time-based train/test split
# Add time-based features (day of year, month, quarter)
```

### 3. Ensemble Approach
```python
# Weighted voting ensemble
models = {
    'lstm': lstm_model,
    'xgboost': xgb_model,
    'prophet': prophet_model,
    'rf': rf_model
}

# Dynamic weight optimization based on recent performance
weights = optimize_weights(models, validation_data)
```

### 4. Additional Data Sources

#### A. COT (Commitment of Traders) Data
- Track large speculators vs commercials positioning
- Strong predictor of price movements
- Available from CFTC weekly

#### B. Options Flow
- Put/call ratios
- Implied volatility
- Option volumes at key strikes

#### C. Physical Market Data
- Warehouse stocks in key ports
- Shipping rates from West Africa
- Currency pairs (USD/GHS, USD/XOF)

### 5. Model Training Improvements

#### A. Time-Aware Cross-Validation
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, gap=30)  # 30-day gap
```

#### B. Custom Loss Functions
```python
# Asymmetric loss - penalize missing buy signals more
def custom_loss(y_true, y_pred):
    # Higher penalty for false negatives (missing opportunities)
    return K.mean(K.square(y_true - y_pred) * K.where(y_true > y_pred, 2.0, 1.0))
```

#### C. Walk-Forward Analysis
- Train on 12 months, predict 1 month
- Roll forward and retrain monthly
- Track performance decay

### 6. Signal Generation Logic

#### Current (Simple Threshold)
```python
signal = 'buy' if composite_score < 0.35 else 'neutral'
```

#### Improved (Multi-Factor with Confirmation)
```python
def generate_signal(df, row_idx):
    signals = []
    
    # Trend confirmation
    if df.iloc[row_idx]['price_trend'] > df.iloc[row_idx-5:row_idx]['price_trend'].mean():
        signals.append(1)
    
    # Momentum confirmation
    if df.iloc[row_idx]['price_momentum_30d'] > 0.02:  # 2% monthly
        signals.append(1)
    
    # Weather severity
    if df.iloc[row_idx]['rainfall_anomaly'] > df['rainfall_anomaly'].quantile(0.8):
        signals.append(1)
    
    # Trade disruption
    if df.iloc[row_idx]['trade_volume_change'] < -0.15:  # 15% decline
        signals.append(1)
    
    # Require 3/4 signals for buy
    return 'buy' if sum(signals) >= 3 else 'neutral'
```

### 7. Real-Time Adaptation

#### A. Online Learning
```python
from river import ensemble
model = ensemble.AdaptiveRandomForestRegressor()
# Update model with each new data point
```

#### B. Dynamic Feature Selection
- Track feature importance over time
- Drop features that lose predictive power
- Add new features as patterns emerge

### 8. Performance Monitoring

#### A. Key Metrics to Track
- Rolling 3-month accuracy
- Maximum drawdown
- Hit rate on major moves (>10%)
- False positive rate
- Time to signal (lead time)

#### B. A/B Testing Framework
```python
class ModelABTest:
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
        self.performance_a = []
        self.performance_b = []
    
    def evaluate(self, X, y):
        pred_a = self.model_a.predict(X)
        pred_b = self.model_b.predict(X)
        # Track performance and switch to better model
```

## Implementation Priority

1. **Immediate (Week 1)**
   - Add lagged features
   - Implement time-aware cross-validation
   - Add XGBoost model

2. **Short-term (Month 1)**
   - Integrate COT data
   - Build LSTM model
   - Create ensemble framework

3. **Medium-term (Quarter 1)**
   - Add options flow data
   - Implement online learning
   - Deploy A/B testing

## Expected Improvements
- Signal Accuracy: 53.5% → 65-70%
- Sharpe Ratio: 0.69 → 1.2+
- False Positive Rate: Reduce by 40%
- Lead Time: Maintain 1-2 months

## Success Criteria
- Achieve 65%+ accuracy on out-of-sample data
- Generate profitable signals in backtesting
- Provide actionable insights with sufficient lead time
- Maintain low false positive rate (<30%)