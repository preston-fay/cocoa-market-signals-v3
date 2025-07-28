#!/usr/bin/env python3
"""Add sentiment analysis to news articles"""
import sqlite3
from textblob import TextBlob
import pandas as pd

conn = sqlite3.connect("data/cocoa_market_signals_real.db")
cursor = conn.cursor()

# Get articles without sentiment
df = pd.read_sql_query("""
    SELECT id, title, description 
    FROM news_articles 
    WHERE sentiment_score IS NULL
    LIMIT 2000
""", conn)

print(f"📰 Processing sentiment for {len(df)} articles...")

updated = 0
for _, row in df.iterrows():
    try:
        # Combine title and description
        text = f"{row['title']} {row['description'] or ''}"
        
        # Get sentiment
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        
        # Update database
        cursor.execute("""
            UPDATE news_articles 
            SET sentiment_score = ?
            WHERE id = ?
        """, (sentiment, row['id']))
        
        updated += 1
        
        if updated % 100 == 0:
            conn.commit()
            print(f"   Processed {updated}/{len(df)}...")
            
    except Exception as e:
        pass

conn.commit()
conn.close()

print(f"✅ Updated sentiment for {updated} articles")

# Verify sentiment distribution
conn = sqlite3.connect("data/cocoa_market_signals_real.db")
result = conn.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN sentiment_score > 0.1 THEN 1 ELSE 0 END) as positive,
        SUM(CASE WHEN sentiment_score < -0.1 THEN 1 ELSE 0 END) as negative,
        SUM(CASE WHEN sentiment_score BETWEEN -0.1 AND 0.1 THEN 1 ELSE 0 END) as neutral,
        AVG(sentiment_score) as avg_sentiment,
        MIN(sentiment_score) as min_sentiment,
        MAX(sentiment_score) as max_sentiment
    FROM news_articles
    WHERE sentiment_score IS NOT NULL
""").fetchone()

print(f"\n📊 Sentiment Analysis Summary:")
print(f"   Total analyzed: {result[0]}")
print(f"   Positive: {result[1]} ({result[1]/result[0]*100:.1f}%)")
print(f"   Negative: {result[2]} ({result[2]/result[0]*100:.1f}%)")
print(f"   Neutral: {result[3]} ({result[3]/result[0]*100:.1f}%)")
print(f"   Average: {result[4]:.3f}")
print(f"   Range: {result[5]:.3f} to {result[6]:.3f}")

conn.close()