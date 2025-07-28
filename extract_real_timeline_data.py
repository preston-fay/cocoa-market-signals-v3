#!/usr/bin/env python3
"""
Extract REAL timeline data from our predictions
NO FAKE DATA - ONLY ACTUAL RESULTS
"""
import pandas as pd
import json
import sqlite3

# Load predictions
pred_df = pd.read_csv('data/processed/model_predictions_comparison.csv', parse_dates=['date'])
pred_df['year_month'] = pred_df['date'].dt.to_period('M')
pred_df['correct'] = (pred_df['actual'] > 0) == (pred_df['xgb'] > 0)

# Group by month for REAL accuracy
monthly_accuracy = pred_df.groupby('year_month').agg({
    'correct': 'mean',
    'actual': ['count', 'std'],
    'xgb': 'count'
}).round(3)

print("REAL Monthly Accuracy:")
print(monthly_accuracy)

# Find REAL significant events (where we predicted large moves correctly)
significant = pred_df[
    (abs(pred_df['actual']) > 0.05) & 
    pred_df['correct']
].sort_values('date')

print(f"\nREAL Significant Events: {len(significant)}")

# Get weather and news context for these REAL events
conn = sqlite3.connect("data/cocoa_market_signals_real.db")

timeline_data = []
for period in pred_df['year_month'].unique():
    month_data = pred_df[pred_df['year_month'] == period]
    
    # Calculate REAL metrics
    accuracy = (month_data['correct'].sum() / len(month_data) * 100)
    volatility = month_data['actual'].std()
    
    # Get REAL news count for this month
    month_str = str(period)
    news_count = pd.read_sql_query(f"""
        SELECT COUNT(*) as count 
        FROM news_articles 
        WHERE strftime('%Y-%m', published_date) = '{month_str}'
    """, conn).iloc[0]['count']
    
    # Determine predictability based on REAL accuracy
    if accuracy >= 85:
        predictability = 'high'
    elif accuracy >= 70:
        predictability = 'medium'
    else:
        predictability = 'low'
    
    # Find the most significant REAL event this month
    month_significant = month_data[abs(month_data['actual']) > 0.03]
    if len(month_significant) > 0:
        biggest = month_significant.loc[month_significant['actual'].abs().idxmax()]
        event_desc = f"{biggest['actual']:+.1%} move on {biggest['date'].strftime('%Y-%m-%d')}"
    else:
        event_desc = f"{news_count} news articles, volatility: {volatility:.3f}"
    
    timeline_data.append({
        'date': month_str,
        'predictability': predictability,
        'accuracy': round(accuracy),
        'events': event_desc,
        'significant': len(month_significant) > 0,
        'prediction_count': int(len(month_data)),
        'news_count': int(news_count)
    })

# Sort timeline data chronologically
timeline_data.sort(key=lambda x: x['date'])

# Save REAL timeline data
with open('data/processed/real_timeline_data.json', 'w') as f:
    json.dump(timeline_data, f, indent=2)

print(f"\nSaved {len(timeline_data)} months of REAL timeline data")

# Also save the REAL significant events with full details
significant_events = []
for _, event in significant.iterrows():
    significant_events.append({
        'date': event['date'].isoformat(),
        'actual_return': float(event['actual']),
        'predicted_return': float(event['xgb']),
        'accuracy': 'correct'
    })

with open('data/processed/real_significant_events.json', 'w') as f:
    json.dump(significant_events, f, indent=2)

print(f"Saved {len(significant_events)} REAL significant events")

conn.close()