#!/usr/bin/env python3
"""
Extract detailed event data including triggering factors
"""
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def get_detailed_events():
    """Extract events with all triggering data"""
    conn = sqlite3.connect('data/cocoa_market_signals_real.db')
    
    # Get significant price movements
    price_query = """
    SELECT 
        p1.date,
        p1.close as price,
        p2.close as prev_price,
        (p1.close - p2.close) / p2.close as return,
        p1.volume
    FROM price_data p1
    JOIN price_data p2 ON date(p2.date) = date(p1.date, '-1 day')
    WHERE ABS((p1.close - p2.close) / p2.close) > 0.05
    ORDER BY p1.date
    """
    
    significant_moves = pd.read_sql(price_query, conn)
    print(f"Found {len(significant_moves)} significant price movements (>5%)")
    
    detailed_events = []
    
    for _, move in significant_moves.iterrows():
        event_date = move['date']
        event = {
            'date': event_date,
            'price_change': move['return'],
            'price': move['price'],
            'previous_price': move['prev_price'],
            'volume': move['volume'],
            'news_triggers': [],
            'weather_triggers': [],
            'trade_triggers': []
        }
        
        # Get news around this date (3 days before to 1 day after)
        news_query = """
        SELECT title, description, source, sentiment_score, published_date
        FROM news_articles
        WHERE date(published_date) BETWEEN date(?, '-3 days') AND date(?, '+1 day')
        ORDER BY ABS(sentiment_score) DESC
        LIMIT 5
        """
        news_df = pd.read_sql(news_query, conn, params=[event_date, event_date])
        
        for _, article in news_df.iterrows():
            event['news_triggers'].append({
                'date': article['published_date'],
                'title': article['title'],
                'description': article['description'],
                'source': article['source'],
                'sentiment': article['sentiment_score']
            })
        
        # Get weather anomalies around this date
        weather_query = """
        SELECT 
            date,
            location,
            temp_mean,
            rainfall,
            30 as avg_temp,  -- Approximate average for tropical regions
            100 as avg_rainfall  -- Approximate average monthly rainfall
        FROM weather_data
        WHERE date BETWEEN date(?, '-7 days') AND date(?)
        AND (
            ABS(temp_mean - 30) > 3 OR
            ABS(rainfall - 5) > 10  -- Daily rainfall anomaly
        )
        ORDER BY date DESC
        LIMIT 5
        """
        weather_df = pd.read_sql(weather_query, conn, params=[event_date, event_date])
        
        for _, weather in weather_df.iterrows():
            event['weather_triggers'].append({
                'date': weather['date'],
                'location': weather['location'],
                'temp_anomaly': weather['temp_mean'] - 30,
                'rainfall_anomaly': weather['rainfall'] - 5,
                'temperature': weather['temp_mean'],
                'rainfall': weather['rainfall']
            })
        
        # Get trade data around this date
        trade_query = """
        SELECT 
            year,
            month,
            SUM(trade_value_usd) as total_value,
            SUM(quantity_kg) as total_volume
        FROM comtrade_data
        WHERE year = ? AND month = ?
        GROUP BY year, month
        """
        # Extract year and month from event date
        event_dt = datetime.strptime(event_date[:10], '%Y-%m-%d')
        trade_df = pd.read_sql(trade_query, conn, params=[event_dt.year, event_dt.month])
        
        for _, trade in trade_df.iterrows():
            event['trade_triggers'].append({
                'period': f"{int(trade['year'])}-{int(trade['month']):02d}",
                'trade_value': trade['total_value'],
                'volume_kg': trade['total_volume']
            })
        
        # Determine primary trigger type
        if event['weather_triggers'] and abs(event['weather_triggers'][0]['temp_anomaly']) > 3:
            event['primary_trigger'] = 'weather'
            event['trigger_description'] = f"Temperature anomaly: {event['weather_triggers'][0]['temp_anomaly']:.1f}°C"
        elif event['news_triggers'] and abs(event['news_triggers'][0]['sentiment']) > 0.5:
            event['primary_trigger'] = 'news'
            event['trigger_description'] = event['news_triggers'][0]['title']
        elif abs(move['return']) > 0.1:
            event['primary_trigger'] = 'technical'
            event['trigger_description'] = f"Large price movement: {move['return']*100:.1f}%"
        else:
            event['primary_trigger'] = 'mixed'
            event['trigger_description'] = "Multiple factors"
            
        detailed_events.append(event)
    
    conn.close()
    
    # Sort by date
    detailed_events.sort(key=lambda x: x['date'])
    
    # Save detailed events
    with open('data/processed/detailed_events.json', 'w') as f:
        json.dump(detailed_events, f, indent=2)
    
    print(f"\nSaved {len(detailed_events)} detailed events to detailed_events.json")
    
    # Show summary
    trigger_counts = {}
    for event in detailed_events:
        trigger = event['primary_trigger']
        trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
    
    print("\nPrimary triggers:")
    for trigger, count in trigger_counts.items():
        print(f"  {trigger}: {count} events")
    
    return detailed_events

if __name__ == "__main__":
    events = get_detailed_events()
    
    # Show example event
    if events:
        print("\nExample event:")
        example = events[0]
        print(f"Date: {example['date']}")
        print(f"Price change: {example['price_change']*100:.1f}%")
        print(f"Primary trigger: {example['primary_trigger']}")
        print(f"Description: {example['trigger_description']}")
        print(f"News articles: {len(example['news_triggers'])}")
        print(f"Weather anomalies: {len(example['weather_triggers'])}")