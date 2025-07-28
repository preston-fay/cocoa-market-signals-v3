#!/usr/bin/env python3
"""
Load the full news dataset with 6000+ articles
"""
import pandas as pd
import json
import os

def load_full_news():
    """Load all news data from various sources"""
    print("📰 Loading full news dataset...")
    
    all_articles = []
    
    # Check all possible news data locations
    news_files = [
        'data/processed/cocoa_news_sentiment.csv',
        'data/processed/all_cocoa_news.csv',
        'data/processed/gdelt_cocoa_news.json',
        'data/historical/news/gdelt_cocoa_articles.json',
        'data/processed/comprehensive_news_with_sentiment.csv'
    ]
    
    for file in news_files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
            try:
                if file.endswith('.csv'):
                    df = pd.read_csv(file)
                    print(f"   Loaded {len(df)} articles")
                    # Convert to dict format
                    for _, row in df.iterrows():
                        article = {
                            'title': row.get('title', 'No title'),
                            'publishedAt': str(row.get('published_date', row.get('publishedAt', '2024-01-01'))),
                            'sentiment_score': row.get('sentiment_score', 0),
                            'source': {'name': row.get('source', 'Unknown')}
                        }
                        all_articles.append(article)
                        
                elif file.endswith('.json'):
                    with open(file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_articles.extend(data)
                        print(f"   Loaded {len(data)} articles")
                        
            except Exception as e:
                print(f"   ❌ Error loading {file}: {e}")
    
    print(f"\n📊 Total articles loaded: {len(all_articles)}")
    
    # Save consolidated news data
    if all_articles:
        output_file = 'data/processed/consolidated_news_with_sentiment.json'
        with open(output_file, 'w') as f:
            json.dump(all_articles, f, indent=2)
        print(f"💾 Saved to: {output_file}")
        
        # Also create a minimal version for the current dashboard
        minimal_news = all_articles[:100]  # First 100 for quick loading
        with open('data/historical/news/real_cocoa_news.json', 'w') as f:
            json.dump(minimal_news, f, indent=2)
        print(f"💾 Updated real_cocoa_news.json with {len(minimal_news)} articles")
    
    return len(all_articles)

if __name__ == "__main__":
    count = load_full_news()
    if count == 0:
        print("\n⚠️ No news data found! The dashboard won't show news-triggered signals.")
        print("💡 To fix: Re-run the news collection scripts from earlier")