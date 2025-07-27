#!/usr/bin/env python3
"""
Fetch more cocoa news from GDELT with multiple queries
"""
import requests
import json
from datetime import datetime, timedelta
from sqlmodel import Session
from app.core.database import engine
from app.models.news_article import NewsArticle
import time
import urllib.parse

def fetch_gdelt_batch(query: str, timespan: str = "3months") -> list:
    """Fetch news from GDELT for a specific query"""
    
    # URL encode the query
    encoded_query = urllib.parse.quote(query)
    
    # GDELT DOC API v2
    url = f"https://api.gdeltproject.org/api/v2/doc/doc"
    
    params = {
        'query': query,
        'mode': 'artlist',
        'format': 'json',
        'maxrecords': '250',  # Max allowed
        'timespan': timespan,
        'sort': 'hybridrel'
    }
    
    try:
        print(f"  Searching: {query}")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            print(f"    Found {len(articles)} articles")
            return articles
        else:
            print(f"    Error: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"    Error: {str(e)}")
        return []

def save_articles_to_db(articles: list, source_query: str):
    """Save articles to database"""
    
    with Session(engine) as session:
        saved = 0
        
        for article in articles:
            # Skip if no URL
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
                # Format: YYYYMMDDTHHMMSSZ
                published_date = datetime.strptime(seendate[:15], '%Y%m%dT%H%M%S')
            except:
                published_date = datetime.now()
            
            # Extract content
            title = article.get('title', '')[:500]
            
            # Create article
            news_article = NewsArticle(
                published_date=published_date,
                fetched_date=datetime.now(),
                url=url,
                title=title,
                content=title,  # GDELT doesn't provide full content
                summary=title,
                source=article.get('domain', 'Unknown'),
                source_type='gdelt_api',
                relevance_score=0.7,
                processed=False
            )
            
            session.add(news_article)
            saved += 1
        
        session.commit()
        return saved

def main():
    """Fetch news with multiple targeted queries"""
    
    print("🔍 Fetching cocoa news from GDELT...")
    print("=" * 60)
    
    # Multiple specific queries to maximize coverage
    queries = [
        # Price and market queries
        '"cocoa prices" OR "cocoa futures"',
        '"cocoa market" analysis',
        'cocoa commodity trading',
        
        # Production queries
        '"cocoa production" Ghana OR "Ivory Coast"',
        'COCOBOD Ghana cocoa',
        '"Côte d\'Ivoire" cocoa export',
        
        # Supply chain
        '"cocoa harvest" 2024 OR 2025',
        '"cocoa shortage" chocolate',
        'cocoa "supply chain" disruption',
        
        # Weather and disease
        'cocoa weather drought OR rainfall',
        '"black pod disease" cocoa',
        'cocoa "climate change" impact',
        
        # Industry news
        'Barry Callebaut OR Cargill cocoa',
        'Nestle OR Mars cocoa sustainability',
        'cocoa "living income differential"'
    ]
    
    total_saved = 0
    all_articles = []
    
    for query in queries:
        articles = fetch_gdelt_batch(query, timespan="6months")
        if articles:
            saved = save_articles_to_db(articles, query)
            total_saved += saved
            print(f"    Saved {saved} new articles")
        
        # Rate limiting
        time.sleep(2)
    
    print(f"\n✅ Total new articles saved: {total_saved}")
    
    # Check total in database
    from sqlmodel import select, func
    with Session(engine) as session:
        total = session.scalar(select(func.count(NewsArticle.id)))
        sources = session.scalar(select(func.count(func.distinct(NewsArticle.source))))
        
        print(f"\n📊 Database Status:")
        print(f"  Total articles: {total}")
        print(f"  Unique sources: {sources}")

if __name__ == "__main__":
    from sqlmodel import select
    main()