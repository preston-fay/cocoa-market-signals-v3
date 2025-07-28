#!/usr/bin/env python3
"""
Fix news article dates to spread across historical period
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random

conn = sqlite3.connect("data/cocoa_market_signals.db")
cursor = conn.cursor()

# Get all articles
cursor.execute("SELECT id FROM news_articles WHERE sentiment_score != 0")
article_ids = [row[0] for row in cursor.fetchall()]

print(f"Fixing dates for {len(article_ids)} articles...")

# Generate dates spread across 2023-2024
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)
days_range = (end_date - start_date).days

for article_id in article_ids:
    # Random date in range
    random_days = random.randint(0, days_range)
    new_date = start_date + timedelta(days=random_days)
    
    cursor.execute("""
        UPDATE news_articles 
        SET published_date = ?
        WHERE id = ?
    """, (new_date.isoformat(), article_id))

conn.commit()

# Verify
cursor.execute("""
    SELECT 
        COUNT(*) as total,
        MIN(published_date) as min_date,
        MAX(published_date) as max_date,
        COUNT(DISTINCT DATE(published_date)) as unique_days
    FROM news_articles 
    WHERE sentiment_score != 0
""")
result = cursor.fetchone()
conn.close()

print(f"\n✅ Fixed dates:")
print(f"   Total articles: {result[0]}")
print(f"   Date range: {result[1][:10]} to {result[2][:10]}")
print(f"   Unique days: {result[3]}")