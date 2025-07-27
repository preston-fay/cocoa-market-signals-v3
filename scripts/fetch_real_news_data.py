#!/usr/bin/env python3
"""
Fetch REAL news data - NO FAKE DATA
"""
import requests
import json
from datetime import datetime, timedelta
import feedparser
import time
from pathlib import Path

def fetch_real_news():
    """
    Fetch REAL news from public sources
    """
    print("\nFetching REAL news data - NO FAKE DATA")
    print("="*60)
    
    all_articles = []
    
    # 1. Try GDELT API (free, no auth required)
    print("\n1. Trying GDELT Project API (free, public)...")
    gdelt_articles = fetch_gdelt_news()
    if gdelt_articles:
        all_articles.extend(gdelt_articles)
    
    # 2. Try public RSS feeds that work
    print("\n2. Trying public RSS feeds...")
    
    # Financial news RSS feeds
    rss_feeds = [
        {
            "name": "MarketWatch Commodities",
            "url": "http://feeds.marketwatch.com/marketwatch/marketpulse/"
        },
        {
            "name": "Investing.com Commodities", 
            "url": "https://www.investing.com/rss/commodities.rss"
        },
        {
            "name": "FT Commodities",
            "url": "https://www.ft.com/commodities?format=rss"
        }
    ]
    
    for feed in rss_feeds:
        print(f"\n   Trying {feed['name']}...")
        articles = fetch_rss_feed(feed['url'], feed['name'])
        if articles:
            all_articles.extend(articles)
            print(f"   ✓ Got {len(articles)} articles")
        else:
            print(f"   ❌ Failed")
    
    # 3. Try News API with free tier
    print("\n3. For production, consider these APIs:")
    print("   - NewsAPI.org (free tier: 100 requests/day)")
    print("   - GNews API (free tier: 100 requests/day)")
    print("   - Bing News Search API (free tier available)")
    print("   - Alpha Vantage News Sentiment (free with API key)")
    
    if all_articles:
        save_real_news(all_articles)
    else:
        print("\n❌ No news data fetched. Checking for alternatives...")
        check_existing_options()

def fetch_gdelt_news():
    """
    Fetch from GDELT Project (real-time global news)
    """
    try:
        # GDELT API for cocoa-related news
        query = "cocoa AND (price OR harvest OR production OR Ghana OR Ivory Coast)"
        url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={query}&mode=artlist&format=json&maxrecords=50"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            articles = []
            
            if 'articles' in data:
                for article in data['articles']:
                    articles.append({
                        "date": article.get('seendate', ''),
                        "headline": article.get('title', ''),
                        "url": article.get('url', ''),
                        "source": article.get('domain', ''),
                        "source_api": "GDELT"
                    })
            
            print(f"   ✓ GDELT: Found {len(articles)} cocoa-related articles")
            return articles
            
    except Exception as e:
        print(f"   ❌ GDELT error: {str(e)}")
    
    return []

def fetch_rss_feed(url, source_name):
    """
    Fetch articles from RSS feed
    """
    try:
        feed = feedparser.parse(url)
        articles = []
        
        # Look for cocoa-related articles
        cocoa_keywords = ['cocoa', 'chocolate', 'ghana', 'ivory coast', 'cacao']
        
        for entry in feed.entries[:50]:  # Last 50 entries
            title = entry.get('title', '').lower()
            summary = entry.get('summary', '').lower()
            
            # Check if cocoa-related
            if any(keyword in title or keyword in summary for keyword in cocoa_keywords):
                articles.append({
                    "date": entry.get('published', ''),
                    "headline": entry.get('title', ''),
                    "url": entry.get('link', ''),
                    "summary": entry.get('summary', '')[:200],
                    "source": source_name,
                    "source_api": "RSS"
                })
        
        return articles
        
    except Exception as e:
        return []

def save_real_news(articles):
    """
    Save real news data
    """
    output_dir = Path("data/historical/news")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "real_cocoa_news.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "source": "REAL news from public APIs",
            "fetch_date": datetime.now().isoformat(),
            "total_articles": len(articles),
            "articles": articles
        }, f, indent=2)
    
    print(f"\n✓ Saved {len(articles)} REAL news articles to {output_file}")

def check_existing_options():
    """
    Check what we can do without fake data
    """
    print("\n" + "="*60)
    print("OPTIONS FOR REAL NEWS DATA:")
    print("="*60)
    
    print("\n1. FREE APIs (require signup):")
    print("   - NewsAPI.org: Sign up at https://newsapi.org/register")
    print("   - GNews.io: Sign up at https://gnews.io/register")
    print("   - Alpha Vantage: Get key at https://www.alphavantage.co/support/")
    
    print("\n2. Set environment variable with API key:")
    print("   export NEWS_API_KEY='your-key-here'")
    
    print("\n3. CommonCrawl approach:")
    print("   - Download CommonCrawl index for news sites")
    print("   - Extract cocoa-related articles")
    print("   - Requires more setup but 100% free")
    
    print("\n4. Manual collection:")
    print("   - Manually collect headlines from:")
    print("     • https://www.reuters.com/markets/commodities/")
    print("     • https://www.bloomberg.com/markets/commodities")
    print("     • https://www.ft.com/commodities")

if __name__ == "__main__":
    fetch_real_news()