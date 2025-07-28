#!/usr/bin/env python3
"""
Verify what data we ACTUALLY have vs what we claimed
"""
import pandas as pd
import json
import os

print("🔍 VERIFYING ACTUAL DATA vs CLAIMS")
print("=" * 60)

# 1. Check news data
print("\n1. NEWS DATA:")
print("-" * 30)

news_count = 0
if os.path.exists('data/historical/news/real_cocoa_news.json'):
    with open('data/historical/news/real_cocoa_news.json', 'r') as f:
        news = json.load(f)
        news_count = len(news)
        print(f"✅ Found: {news_count} articles in real_cocoa_news.json")
        
        # Check if they have sentiment
        with_sentiment = sum(1 for n in news if 'sentiment_score' in n)
        print(f"   With sentiment: {with_sentiment}")
else:
    print("❌ No news file found")

print(f"\n🚨 CLAIMED: 6,229 articles")
print(f"🚨 ACTUAL: {news_count} articles")
print(f"🚨 MISSING: {6229 - news_count} articles!")

# 2. Check weather data
print("\n2. WEATHER DATA:")
print("-" * 30)

weather_files = [
    'data/historical/weather/weather_data_2yr.csv',
    'data/processed/weather_features.csv'
]

total_weather = 0
for file in weather_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        total_weather += len(df)
        print(f"✅ {file}: {len(df)} records")

print(f"\n🚨 CLAIMED: 18,431 weather records")
print(f"🚨 ACTUAL: {total_weather} records")

# 3. Check predictions/backtesting
print("\n3. PREDICTIONS DATA:")
print("-" * 30)

if os.path.exists('data/processed/backtesting_results_full.csv'):
    df = pd.read_csv('data/processed/backtesting_results_full.csv')
    print(f"✅ Backtesting results: {len(df)} predictions")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
else:
    print("❌ No backtesting results found")

# 4. Check comprehensive dataset
print("\n4. COMPREHENSIVE DATASET:")
print("-" * 30)

if os.path.exists('data/processed/comprehensive_train.csv'):
    train = pd.read_csv('data/processed/comprehensive_train.csv')
    test = pd.read_csv('data/processed/comprehensive_test.csv')
    print(f"✅ Training: {len(train)} samples")
    print(f"✅ Test: {len(test)} samples")
    print(f"   Total: {len(train) + len(test)} samples")
    
    # Check if news features exist
    news_features = [col for col in train.columns if 'sentiment' in col or 'news' in col]
    print(f"   News features: {news_features}")
else:
    print("❌ No comprehensive dataset found")

print("\n" + "="*60)
print("💡 CONCLUSION:")
print("The agents collected data but FAILED to:")
print("1. Save the GDELT news articles properly")
print("2. Integrate news into the feature extraction")
print("3. Use the full date range in predictions")
print("4. Validate that collected data was actually saved")
print("\nThis is why there are NO news-triggered signals!")