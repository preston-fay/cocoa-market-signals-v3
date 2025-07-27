#!/usr/bin/env python3
"""
Targeted historical news collection with specific date parameters
"""
import requests
import json
from datetime import datetime, timedelta
from sqlmodel import Session, select, func
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.core.database import engine
from app.models.news_article import NewsArticle
import time

def collect_historical_gdelt():
    """Collect historical news using GDELT's date parameters"""
    
    session = Session(engine)
    articles_saved = 0
    
    print("🎯 Targeted Historical Collection")
    print("=" * 60)
    
    # Try different approaches for historical data
    
    # Approach 1: Use specific date ranges in GDELT
    print("\n1️⃣ Using GDELT date range parameters...")
    
    # Important historical periods
    date_ranges = [
        # 2024 Price surge period
        ("20240201", "20240430", "2024 Q1 Price Surge"),
        ("20240501", "20240731", "2024 Q2"),
        ("20240801", "20241031", "2024 Q3"),
        ("20231001", "20231231", "2023 Q4 Harvest"),
        ("20230701", "20230930", "2023 Q3"),
    ]
    
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    for start_date, end_date, period_name in date_ranges:
        print(f"\n📅 {period_name}")
        
        # Multiple specific queries
        queries = [
            'cocoa',
            'cocoa price',
            'Ghana cocoa',
            'Ivory Coast cocoa', 
            'cocoa harvest',
            'cocoa market',
            'COCOBOD',
            'cocoa futures'
        ]
        
        for query in queries:
            params = {
                'query': query,
                'mode': 'artlist',
                'format': 'json',
                'maxrecords': '250',
                'startdatetime': start_date + '000000',
                'enddatetime': end_date + '235959',
                'sort': 'hybridrel'
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    if articles:
                        # Save articles
                        saved = 0
                        for article in articles:
                            url = article.get('url', '').strip()
                            if not url:
                                continue
                            
                            # Check if exists
                            existing = session.exec(
                                select(NewsArticle).where(NewsArticle.url == url)
                            ).first()
                            
                            if existing:
                                continue
                            
                            # Parse date
                            seendate = article.get('seendate', '')
                            try:
                                published_date = datetime.strptime(seendate[:15], '%Y%m%dT%H%M%S')
                            except:
                                continue
                            
                            # Create article
                            news_article = NewsArticle(
                                published_date=published_date,
                                fetched_date=datetime.now(),
                                url=url,
                                title=article.get('title', '')[:500],
                                content=article.get('title', ''),
                                summary=article.get('title', '')[:500],
                                source=article.get('domain', 'Unknown'),
                                source_type=f'historical_{period_name.replace(" ", "_")}',
                                relevance_score=0.7,
                                processed=False
                            )
                            
                            session.add(news_article)
                            saved += 1
                            articles_saved += 1
                        
                        session.commit()
                        
                        if saved > 0:
                            print(f"  '{query}': {saved} articles")
                
            except Exception as e:
                print(f"  Error with '{query}': {str(e)}")
            
            time.sleep(1)  # Rate limit
    
    # Approach 2: Try using GDELT's theme codes
    print("\n2️⃣ Using GDELT theme codes for commodities...")
    
    theme_queries = [
        'theme:COMMODITY_MARKETS cocoa',
        'theme:ECON_PRICE cocoa',
        'theme:ENV_AGRO cocoa',
        'theme:WEATHER cocoa africa'
    ]
    
    for query in theme_queries:
        params = {
            'query': query,
            'mode': 'artlist',
            'format': 'json',
            'maxrecords': '250',
            'timespan': '2years',
            'sort': 'hybridrel'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                print(f"  Theme '{query}': {len(articles)} results")
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Historical collection complete!")
    print(f"   New articles saved: {articles_saved}")
    
    # Check coverage
    historical_count = session.exec(
        select(func.count(NewsArticle.id))
        .where(NewsArticle.published_date < datetime(2025, 1, 1))
    ).one()
    
    print(f"   Total historical articles (pre-2025): {historical_count}")
    
    # Show date range
    oldest = session.scalar(select(func.min(NewsArticle.published_date)))
    newest = session.scalar(select(func.max(NewsArticle.published_date)))
    print(f"   Full date range: {oldest.date()} to {newest.date()}")
    
    session.close()

if __name__ == "__main__":
    collect_historical_gdelt()