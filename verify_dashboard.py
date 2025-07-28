#!/usr/bin/env python3
"""
Verify dashboard data integrity
"""
import json
import pandas as pd
import sqlite3

print("=== DASHBOARD VERIFICATION ===\n")

# 1. Check real timeline data
with open('data/processed/real_timeline_data.json', 'r') as f:
    timeline = json.load(f)
    
print(f"1. Timeline has {len(timeline)} months")
print(f"   First: {timeline[0]['date']} (Accuracy: {timeline[0]['accuracy']}%)")
print(f"   Last: {timeline[-1]['date']} (Accuracy: {timeline[-1]['accuracy']}%)")

# 2. Check predictions file
pred_df = pd.read_csv('data/processed/model_predictions_comparison.csv', parse_dates=['date'])
print(f"\n2. Predictions file has {len(pred_df)} records")
print(f"   Date range: {pred_df['date'].min()} to {pred_df['date'].max()}")

# 3. Check dashboard data
with open('data/processed/real_dashboard_data.json', 'r') as f:
    dash_data = json.load(f)
    
print(f"\n3. Dashboard data summary:")
print(f"   Total predictions: {dash_data['data_summary']['total_predictions']}")
print(f"   Date range: {dash_data['data_summary']['date_range']['start']} to {dash_data['data_summary']['date_range']['end']}")
print(f"   Significant events: {len(dash_data['significant_events'])}")

# 4. Check database
conn = sqlite3.connect("data/cocoa_market_signals_real.db")
price_count = pd.read_sql_query("SELECT COUNT(*) as count FROM price_data", conn).iloc[0]['count']
news_count = pd.read_sql_query("SELECT COUNT(*) as count FROM news_articles", conn).iloc[0]['count']
conn.close()

print(f"\n4. Database records:")
print(f"   Price records: {price_count}")
print(f"   News articles: {news_count}")

# 5. Verify Actual vs Predicted alignment
print(f"\n5. Checking Actual vs Predicted data:")
sample = pred_df.head(5)
for _, row in sample.iterrows():
    print(f"   {row['date'].strftime('%Y-%m-%d')}: Actual={row['actual']:+.3f}, XGB={row['xgb']:+.3f}, Correct={row['actual'] * row['xgb'] > 0}")

print("\n=== VERIFICATION COMPLETE ===")