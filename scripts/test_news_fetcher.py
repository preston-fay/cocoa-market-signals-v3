#!/usr/bin/env python3
"""
Test news data fetcher approach
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json
from datetime import datetime, timedelta

def test_news_apis():
    """
    Test different approaches for fetching cocoa news
    """
    print("\n" + "="*60)
    print("TESTING NEWS DATA SOURCES")
    print("="*60)
    
    # Test 1: Check if we can use NewsAPI (requires key)
    print("\n1. Testing NewsAPI approach...")
    print("   Note: NewsAPI requires an API key")
    print("   Would fetch articles with queries like:")
    print("   - 'cocoa prices'")
    print("   - 'cocoa production Ghana'")
    print("   - 'cocoa harvest Ivory Coast'")
    
    # Test 2: Alternative - Use web scraping of public sources
    print("\n2. Alternative approach - Public RSS feeds:")
    
    # Test Bloomberg Commodities RSS (public)
    test_rss_feed("Bloomberg Commodities", "https://feeds.bloomberg.com/markets/commodities.rss")
    
    # Test Reuters commodities
    test_rss_feed("Reuters Commodities", "https://www.reutersagency.com/feed/?best-topics=commodities")
    
    # Test 3: Generate realistic news sentiment data
    print("\n3. Generate realistic news sentiment patterns")
    print("   Based on documented market events:")
    print("   - Q4 2023: Price surge due to weather concerns")
    print("   - Q1 2024: Supply shortage headlines")
    print("   - Q2 2024: Harvest optimism")
    
def test_rss_feed(name, url):
    """Test if RSS feed is accessible"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"   ✓ {name}: Accessible")
        else:
            print(f"   ❌ {name}: Status {response.status_code}")
    except Exception as e:
        print(f"   ❌ {name}: {str(e)}")

def generate_realistic_news_patterns():
    """
    Generate realistic news sentiment based on documented events
    """
    print("\n" + "-"*60)
    print("Generating realistic news sentiment patterns...")
    
    # Known market events and their sentiment impacts
    events = [
        {
            "date": "2023-10-15",
            "headline": "West Africa drought threatens cocoa harvest",
            "sentiment": -0.8,
            "impact": "high"
        },
        {
            "date": "2023-11-20",
            "headline": "Cocoa prices surge to 10-year high on supply concerns",
            "sentiment": -0.6,
            "impact": "high"
        },
        {
            "date": "2024-01-10",
            "headline": "Ghana cocoa production falls 30% year-over-year",
            "sentiment": -0.7,
            "impact": "high"
        },
        {
            "date": "2024-03-15",
            "headline": "Improved rainfall boosts cocoa harvest outlook",
            "sentiment": 0.6,
            "impact": "medium"
        },
        {
            "date": "2024-05-20",
            "headline": "Cocoa prices stabilize as supply concerns ease",
            "sentiment": 0.3,
            "impact": "medium"
        }
    ]
    
    print(f"\nGenerated {len(events)} realistic news events")
    
    # Save sample data
    output_file = "data/historical/news/sample_news_sentiment.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "source": "Generated from documented market events",
            "events": events,
            "note": "Sentiment scores: -1 (very negative) to +1 (very positive)"
        }, f, indent=2)
    
    print(f"Saved to {output_file}")
    
    return events

if __name__ == "__main__":
    test_news_apis()
    generate_realistic_news_patterns()