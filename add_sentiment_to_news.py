#!/usr/bin/env python3
"""
Add sentiment analysis to saved news articles
"""
import sqlite3
from textblob import TextBlob
import pandas as pd

def add_sentiment():
    """Add sentiment scores to news articles"""
    conn = sqlite3.connect("data/cocoa_market_signals.db")
    
    # Get articles without sentiment
    df = pd.read_sql_query("""
        SELECT id, title, description 
        FROM news_articles 
        WHERE sentiment_score = 0 OR sentiment_score IS NULL
    """, conn)
    
    print(f"📰 Processing sentiment for {len(df)} articles...")
    
    cursor = conn.cursor()
    updated = 0
    
    for _, row in df.iterrows():
        try:
            # Combine title and description for analysis
            text = f"{row['title']} {row['description'] or ''}"
            
            # Get sentiment
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Update database
            cursor.execute("""
                UPDATE news_articles 
                SET sentiment_score = ?, sentiment_subjectivity = ?
                WHERE id = ?
            """, (sentiment, subjectivity, row['id']))
            
            updated += 1
            
            if updated % 100 == 0:
                conn.commit()
                print(f"   Processed {updated}/{len(df)}...")
                
        except Exception as e:
            print(f"   Error on article {row['id']}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Updated sentiment for {updated} articles")
    
    # Verify
    conn = sqlite3.connect("data/cocoa_market_signals.db")
    df_check = pd.read_sql_query("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN sentiment_score > 0.1 THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN sentiment_score < -0.1 THEN 1 ELSE 0 END) as negative,
            AVG(sentiment_score) as avg_sentiment
        FROM news_articles
    """, conn)
    conn.close()
    
    print("\n📊 Sentiment Summary:")
    print(f"   Positive: {df_check['positive'].values[0]}")
    print(f"   Negative: {df_check['negative'].values[0]}")
    print(f"   Average: {df_check['avg_sentiment'].values[0]:.3f}")

if __name__ == "__main__":
    add_sentiment()