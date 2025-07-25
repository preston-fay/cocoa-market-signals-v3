"""Debug dashboard data loading"""
import pandas as pd
import json
from datetime import datetime

# Test loading data
try:
    df = pd.read_csv("data/processed/unified_real_data.csv")
    print(f"Loaded {len(df)} rows")
    print(f"Date column type: {df['date'].dtype}")
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    print(f"After parsing: {df['date'].dtype}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Get months
    df['year_month'] = df['date'].dt.to_period('M')
    months = df['year_month'].unique()
    print(f"\nAvailable months: {len(months)}")
    print(f"First few: {sorted([str(m) for m in months])[:5]}")
    
    # Test backtest results
    with open("data/processed/backtest_results.json", "r") as f:
        backtest_results = json.load(f)
    print(f"\nBacktest results keys: {list(backtest_results.keys())}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()