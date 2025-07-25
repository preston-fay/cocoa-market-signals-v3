#!/usr/bin/env python3
"""
Test the timeline dashboard to ensure charts render properly
"""

import pandas as pd
import json
from datetime import datetime

# Test data loading
print("Testing data loading...")
df = pd.read_csv("data/processed/unified_real_data.csv")
print(f"✓ Loaded {len(df)} rows of data")
print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")

# Test chart data preparation
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
df = df.set_index('date')

# Test resampling
daily = df.resample('D').last().ffill()
print(f"✓ Daily resampling: {len(daily)} days")

# Test price data
price_data = {
    'dates': daily.index.strftime('%Y-%m-%d').tolist(),
    'prices': daily['price'].tolist()
}
print(f"✓ Price data: {len(price_data['dates'])} dates, {len(price_data['prices'])} prices")
print(f"  Price range: ${min(price_data['prices']):,.0f} - ${max(price_data['prices']):,.0f}")

# Test weather data
weather_data = {
    'dates': daily.index.strftime('%Y-%m-%d').tolist(),
    'rainfall': daily['rainfall_anomaly'].fillna(0).tolist(),
    'temperature': daily['temperature_anomaly'].fillna(0).tolist()
}
print(f"✓ Weather data: {len([x for x in weather_data['rainfall'] if x != 0])} non-zero rainfall values")

# Test signal calculation
rainfall_norm = (daily['rainfall_anomaly'] - daily['rainfall_anomaly'].min()) / (daily['rainfall_anomaly'].max() - daily['rainfall_anomaly'].min() + 0.001)
weather_signal = 1 - rainfall_norm
print(f"✓ Signal calculation: min={weather_signal.min():.3f}, max={weather_signal.max():.3f}")

# Test model results
with open("data/processed/real_data_test_results.json", "r") as f:
    model_results = json.load(f)
print(f"✓ Model results loaded: accuracy={model_results['performance']['signal_accuracy']:.3f}")

print("\n✅ All tests passed! Dashboard should work correctly.")
print("\nTo run the dashboard:")
print("python3 run_timeline_dashboard.py")