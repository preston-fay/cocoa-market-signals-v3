"""Test the dashboard with available data"""
import os
os.environ['LOGURU_LEVEL'] = 'ERROR'

from flask import Flask
import pandas as pd
import json
from pathlib import Path

# Create test app
app = Flask(__name__)

# Check what data we have
data_dir = Path("data")
print("Available data files:")
print("=" * 50)

# Check processed data
processed_dir = data_dir / "processed"
if processed_dir.exists():
    for f in processed_dir.glob("*"):
        print(f"✓ {f.name}")
else:
    print("✗ No processed data found")

# Check historical data
print("\nHistorical data:")
print("-" * 50)

# Prices
price_file = data_dir / "historical/prices/cocoa_daily_prices_2yr.csv"
if price_file.exists():
    df = pd.read_csv(price_file, index_col=0, parse_dates=True)
    print(f"✓ Daily prices: {len(df)} days, ${df['cocoa_cc_close'].iloc[-1]:,.0f} latest")
    
    # Create minimal processed data for dashboard
    processed_dir.mkdir(exist_ok=True)
    
    # Save last 30 days for dashboard
    recent_data = df.tail(30)
    recent_data.to_csv(processed_dir / "recent_prices.csv")
    
    # Create summary
    summary = {
        "last_updated": pd.Timestamp.now().isoformat(),
        "current_price": float(df['cocoa_cc_close'].iloc[-1]),
        "price_change_7d": float(df['cocoa_cc_close'].pct_change(7).iloc[-1] * 100),
        "price_change_30d": float(df['cocoa_cc_close'].pct_change(30).iloc[-1] * 100),
        "volatility": float(df['cocoa_cc_volatility_30d'].iloc[-1]),
        "rsi": float(df['cocoa_cc_rsi'].iloc[-1])
    }
    
    with open(processed_dir / "dashboard_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Created dashboard summary")

# Weather
weather_file = data_dir / "historical/weather/weather_summary_2yr.json"
if weather_file.exists():
    with open(weather_file, 'r') as f:
        weather = json.load(f)
    print(f"✓ Weather data: {len(weather['locations'])} locations")

# Economics
econ_file = data_dir / "historical/economics/inflation_currency_data.json"
if econ_file.exists():
    print(f"✓ Economic data available")

print("\nReady to start dashboard!")
print("Run: python3 run_dashboard.py")