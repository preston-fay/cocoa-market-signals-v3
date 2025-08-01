#!/usr/bin/env python3
"""
Create enhanced timeline data with proper event positioning
"""
import json
import sqlite3
import pandas as pd

# Load detailed events
with open('data/processed/detailed_events.json', 'r') as f:
    events = json.load(f)

# Get all price data
conn = sqlite3.connect('data/cocoa_market_signals_real.db')
query = """
SELECT 
    date,
    close as price,
    LAG(close) OVER (ORDER BY date) as prev_price
FROM price_data
ORDER BY date
"""
price_df = pd.read_sql(query, conn, parse_dates=['date'])
conn.close()

# Calculate returns
price_df['return'] = (price_df['price'] - price_df['prev_price']) / price_df['prev_price']
price_df = price_df.dropna()  # Remove first row with no previous price

# Create a date to return mapping
date_returns = {}
for _, row in price_df.iterrows():
    date_str = row['date'].strftime('%Y-%m-%d')
    date_returns[date_str] = {
        'return': float(row['return']),
        'price': float(row['price']),
        'prev_price': float(row['prev_price'])
    }

# Update events with correct return values
enhanced_events = []
for event in events:
    event_date = event['date']
    
    # Get the actual return from our price data
    if event_date in date_returns:
        actual_return = date_returns[event_date]['return']
        # Verify it matches (approximately)
        if abs(actual_return - event['price_change']) > 0.001:
            print(f"Warning: Return mismatch for {event_date}")
            print(f"  Event says: {event['price_change']:.4f}")
            print(f"  Price data says: {actual_return:.4f}")
        
        event['verified_return'] = actual_return
        event['has_price_data'] = True
    else:
        print(f"No price data for event on {event_date}")
        event['verified_return'] = event['price_change']
        event['has_price_data'] = False
    
    enhanced_events.append(event)

# Save enhanced events
with open('data/processed/enhanced_timeline_events.json', 'w') as f:
    json.dump(enhanced_events, f, indent=2)

# Create timeline data with all prices and properly positioned events
timeline_data = {
    'prices': [],
    'events': enhanced_events
}

# Add all price data
for _, row in price_df.iterrows():
    timeline_data['prices'].append({
        'date': row['date'].strftime('%Y-%m-%d'),
        'price': float(row['price']),
        'return': float(row['return']) * 100  # Convert to percentage
    })

with open('data/processed/timeline_data_complete.json', 'w') as f:
    json.dump(timeline_data, f, indent=2)

print(f"\nCreated enhanced timeline data:")
print(f"  Total prices: {len(timeline_data['prices'])}")
print(f"  Total events: {len(enhanced_events)}")
print(f"  Events with price data: {sum(1 for e in enhanced_events if e['has_price_data'])}")
print(f"  Events without price data: {sum(1 for e in enhanced_events if not e['has_price_data'])}")

# Show date range
print(f"\nDate range: {price_df['date'].min()} to {price_df['date'].max()}")