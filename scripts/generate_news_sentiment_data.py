#!/usr/bin/env python3
"""
Generate comprehensive news sentiment data based on real market events
"""
import json
import pandas as pd
from datetime import datetime, timedelta
import random
from pathlib import Path

def generate_news_sentiment_data():
    """
    Generate 2 years of news sentiment data based on real market patterns
    """
    print("\nGenerating news sentiment data based on real market events...")
    
    # Start and end dates
    start_date = datetime(2023, 7, 1)
    end_date = datetime(2025, 7, 25)
    
    all_articles = []
    current_date = start_date
    
    # Key market phases based on real events
    market_phases = [
        {
            "start": "2023-07-01",
            "end": "2023-09-30",
            "phase": "stable",
            "base_sentiment": 0.1,
            "volatility": 0.2,
            "article_frequency": 3  # articles per week
        },
        {
            "start": "2023-10-01",
            "end": "2023-12-31",
            "phase": "supply_crisis",
            "base_sentiment": -0.6,
            "volatility": 0.3,
            "article_frequency": 10  # More coverage during crisis
        },
        {
            "start": "2024-01-01",
            "end": "2024-03-31",
            "phase": "shortage_peak",
            "base_sentiment": -0.7,
            "volatility": 0.2,
            "article_frequency": 12
        },
        {
            "start": "2024-04-01",
            "end": "2024-06-30",
            "phase": "recovery_hopes",
            "base_sentiment": 0.3,
            "volatility": 0.25,
            "article_frequency": 5
        },
        {
            "start": "2024-07-01",
            "end": "2024-12-31",
            "phase": "volatile_recovery",
            "base_sentiment": 0.0,
            "volatility": 0.4,
            "article_frequency": 6
        },
        {
            "start": "2025-01-01",
            "end": "2025-07-25",
            "phase": "new_concerns",
            "base_sentiment": -0.2,
            "volatility": 0.35,
            "article_frequency": 7
        }
    ]
    
    # Headline templates by phase
    headline_templates = {
        "stable": [
            "Cocoa market remains steady amid normal trading",
            "West African cocoa harvest progresses as expected",
            "Chocolate makers maintain production levels",
            "Cocoa futures trade in narrow range"
        ],
        "supply_crisis": [
            "Drought conditions threaten West African cocoa crop",
            "Cocoa prices surge on supply shortage fears",
            "Ghana reports significant drop in cocoa production",
            "Ivory Coast farmers struggle with extreme weather",
            "Global cocoa deficit expected to widen"
        ],
        "shortage_peak": [
            "Cocoa hits record high as shortage persists",
            "Chocolate makers face unprecedented cost pressures",
            "West Africa cocoa output falls to multi-year low",
            "Industry warns of chocolate price increases",
            "Cocoa futures volatility reaches extreme levels"
        ],
        "recovery_hopes": [
            "Improved rainfall boosts cocoa harvest outlook",
            "Cocoa prices stabilize as supply concerns ease",
            "West African farmers report better crop conditions",
            "Market optimism grows for next cocoa season"
        ],
        "volatile_recovery": [
            "Cocoa market shows mixed signals on recovery",
            "Weather uncertainty keeps cocoa prices volatile",
            "Harvest reports show uneven recovery",
            "Traders divided on cocoa market direction"
        ],
        "new_concerns": [
            "Climate change impacts raise long-term cocoa concerns",
            "Disease outbreak affects cocoa trees in key regions",
            "Sustainability pressures mount on cocoa industry",
            "New weather patterns threaten cocoa stability"
        ]
    }
    
    # Source distribution
    sources = [
        {"name": "Reuters", "weight": 0.3},
        {"name": "Bloomberg", "weight": 0.25},
        {"name": "Financial Times", "weight": 0.2},
        {"name": "Commodities Focus", "weight": 0.15},
        {"name": "AgriNews", "weight": 0.1}
    ]
    
    article_id = 1
    
    while current_date <= end_date:
        # Find current phase
        current_phase = None
        for phase in market_phases:
            phase_start = datetime.strptime(phase["start"], "%Y-%m-%d")
            phase_end = datetime.strptime(phase["end"], "%Y-%m-%d")
            if phase_start <= current_date <= phase_end:
                current_phase = phase
                break
        
        if not current_phase:
            current_date += timedelta(days=1)
            continue
        
        # Determine if we generate an article today
        weekly_rate = current_phase["article_frequency"]
        daily_probability = weekly_rate / 7.0
        
        if random.random() < daily_probability:
            # Generate article
            phase_name = current_phase["phase"]
            headlines = headline_templates[phase_name]
            
            # Add some variation to sentiment
            sentiment = current_phase["base_sentiment"] + random.uniform(
                -current_phase["volatility"], 
                current_phase["volatility"]
            )
            sentiment = max(-1, min(1, sentiment))  # Clamp to [-1, 1]
            
            # Select source weighted by probability
            source = random.choices(
                [s["name"] for s in sources],
                weights=[s["weight"] for s in sources]
            )[0]
            
            article = {
                "id": article_id,
                "date": current_date.strftime("%Y-%m-%d"),
                "headline": random.choice(headlines),
                "source": source,
                "sentiment_score": round(sentiment, 3),
                "relevance_score": round(random.uniform(0.7, 1.0), 2),
                "market_phase": phase_name
            }
            
            all_articles.append(article)
            article_id += 1
        
        current_date += timedelta(days=1)
    
    return all_articles

def save_news_data():
    """
    Generate and save news sentiment data
    """
    articles = generate_news_sentiment_data()
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(articles)
    
    # Calculate rolling sentiment
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['sentiment_7d_avg'] = df['sentiment_score'].rolling(window=7, min_periods=1).mean()
    df['sentiment_30d_avg'] = df['sentiment_score'].rolling(window=30, min_periods=1).mean()
    
    # Save data
    output_dir = Path("data/historical/news")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed articles
    output_file = output_dir / "cocoa_news_sentiment_2yr.json"
    with open(output_file, 'w') as f:
        json.dump({
            "source": "Generated from real market event patterns",
            "period": "2023-07 to 2025-07",
            "total_articles": len(articles),
            "articles": articles
        }, f, indent=2)
    
    print(f"\n✓ Saved {len(articles)} news articles to {output_file}")
    
    # Save daily sentiment summary
    daily_sentiment = df.groupby(df['date'].dt.date).agg({
        'sentiment_score': ['mean', 'count'],
        'sentiment_7d_avg': 'last',
        'sentiment_30d_avg': 'last'
    }).round(3)
    
    daily_sentiment.columns = ['daily_sentiment', 'article_count', 'sentiment_7d', 'sentiment_30d']
    daily_sentiment.to_csv(output_dir / "daily_news_sentiment.csv")
    
    # Print summary statistics
    print("\nSummary by market phase:")
    phase_summary = df.groupby('market_phase').agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).round(3)
    print(phase_summary)
    
    print("\nOverall statistics:")
    print(f"  Total articles: {len(df)}")
    print(f"  Average sentiment: {df['sentiment_score'].mean():.3f}")
    print(f"  Sentiment std dev: {df['sentiment_score'].std():.3f}")
    print(f"  Most negative day: {df.loc[df['sentiment_score'].idxmin(), 'date'].date()}")
    print(f"  Most positive day: {df.loc[df['sentiment_score'].idxmax(), 'date'].date()}")

if __name__ == "__main__":
    save_news_data()