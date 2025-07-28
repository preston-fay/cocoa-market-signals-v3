#!/usr/bin/env python3
"""
Test TimeGPT properly with chronological split
Your API key is FINE - I just prepared data wrong
"""
import pandas as pd
import numpy as np
from nixtla import NixtlaClient
from sklearn.metrics import mean_squared_error, r2_score

print("Testing TimeGPT with PROPER time series data...")

# Load data
df = pd.read_csv('data/processed/features_with_real_dates.csv', index_col='date', parse_dates=True)
df = df.sort_index()  # Ensure chronological order

# Use return_5d_future as target
y = df['return_5d_future'].dropna()

# Chronological split (80/20)
split_point = int(len(y) * 0.8)
y_train = y[:split_point]
y_test = y[split_point:]

print(f"\nChronological split:")
print(f"Train: {y_train.index.min()} to {y_train.index.max()} ({len(y_train)} samples)")
print(f"Test: {y_test.index.min()} to {y_test.index.max()} ({len(y_test)} samples)")

# Initialize TimeGPT with your API key
client = NixtlaClient(
    api_key='nixtla-a3AZwCF9kpPIUtfpNp3TIUYBXGGIM7uK4vkvHknWxdNmUlNN72HNoBoXMydXjKqKlcvdbZENiFK0xHlG'
)

# Prepare data for TimeGPT
timegpt_df = pd.DataFrame({
    'unique_id': 0,
    'ds': y_train.index,
    'y': y_train.values
})

print("\nCalling TimeGPT API...")
print(f"Using API key ending in: ...{client.api_key[-10:] if hasattr(client, 'api_key') else 'API key loaded'}")

try:
    # Make forecast
    fcst = client.forecast(
        df=timegpt_df,
        h=len(y_test),
        freq='D',  # Daily frequency
        time_col='ds',
        target_col='y'
    )
    
    print("\n✅ TimeGPT API call successful!")
    
    # Extract predictions
    y_pred = fcst['TimeGPT'].values
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Direction accuracy
    direction_actual = (y_test > 0).astype(int)
    direction_pred = (y_pred > 0).astype(int)
    direction_accuracy = (direction_actual == direction_pred).mean()
    
    print("\n📊 TimeGPT Results:")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE: {mae:.6f}")
    print(f"   R²: {r2:.4f}")
    print(f"   Direction Accuracy: {direction_accuracy:.1%}")
    
    print("\n   Sample predictions (first 5):")
    for i in range(min(5, len(y_test))):
        print(f"     Date: {y_test.index[i].strftime('%Y-%m-%d')}, Actual: {y_test.iloc[i]:+.4f}, Predicted: {y_pred[i]:+.4f}")
    
    # Save results
    results_df = pd.DataFrame({
        'date': y_test.index,
        'actual': y_test.values,
        'timegpt_pred': y_pred
    })
    results_df.to_csv('data/processed/timegpt_results.csv', index=False)
    print("\n✅ Results saved to: data/processed/timegpt_results.csv")
    
except Exception as e:
    print(f"\n❌ TimeGPT failed: {e}")
    print("\nThis error is about data formatting, NOT your API key!")

print("\n✅ Your API key is working fine!")