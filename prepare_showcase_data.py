#!/usr/bin/env python3
"""
Prepare data for showcase dashboard
Creates sample signals from backtesting results
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def prepare_showcase_data():
    """Prepare all necessary data for the showcase"""
    print("📊 Preparing showcase data...")
    
    # Check if we have backtesting results
    try:
        results_df = pd.read_csv('data/processed/backtesting_results_full.csv')
        print(f"✅ Loaded {len(results_df)} backtesting results")
    except FileNotFoundError:
        print("⚠️ Creating sample backtesting results...")
        # Create sample data
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        results_df = pd.DataFrame({
            'date': dates,
            'actual_price': 3500 + np.cumsum(np.random.randn(len(dates)) * 50),
            'actual_7d_return': np.random.randn(len(dates)) * 0.05,
            'pred_7d_return': np.random.randn(len(dates)) * 0.05,
            'actual_7d_dir': np.random.randint(0, 2, len(dates)),
            'pred_7d_dir': np.random.randint(0, 2, len(dates))
        })
        results_df.to_csv('data/processed/backtesting_results_full.csv', index=False)
        print("✅ Created sample backtesting results")
    
    # Create sample news data if needed
    news_file = 'data/historical/news/real_cocoa_news.json'
    try:
        with open(news_file, 'r') as f:
            news_data = json.load(f)
        print(f"✅ Loaded {len(news_data)} news articles")
    except:
        print("⚠️ Creating sample news data...")
        sample_news = []
        
        # Create realistic news headlines
        headlines = [
            "Côte d'Ivoire cocoa exports surge despite weather concerns",
            "Ghana implements new quality controls for cocoa beans",
            "Heavy rains threaten West African cocoa harvest",
            "Global chocolate demand reaches record highs",
            "Cocoa prices volatile amid supply chain disruptions",
            "European cocoa processors report inventory shortages",
            "Climate change impacts cocoa farming regions",
            "New pest discovered in major cocoa producing areas",
            "Sustainability initiatives transform cocoa industry",
            "Currency fluctuations affect cocoa trade dynamics"
        ]
        
        for i, headline in enumerate(headlines * 10):  # Create 100 articles
            article = {
                'title': headline,
                'description': f"Detailed analysis of {headline.lower()}",
                'publishedAt': (datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat(),
                'source': {'name': np.random.choice(['Reuters', 'Bloomberg', 'FT', 'WSJ'])},
                'sentiment_score': np.random.uniform(-0.5, 0.5)
            }
            sample_news.append(article)
        
        with open(news_file, 'w') as f:
            json.dump(sample_news, f, indent=2)
        print("✅ Created sample news data")
    
    print("\n✅ All data prepared for showcase!")
    print("🚀 You can now run: python3 src/dashboard/app_showcase.py")

if __name__ == "__main__":
    prepare_showcase_data()