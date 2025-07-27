#!/usr/bin/env python3
"""
Load GDELT news articles into database
"""
import json
from datetime import datetime
from sqlmodel import Session
from app.core.database import engine
from app.models.news_article import NewsArticle
from pathlib import Path

def load_gdelt_news():
    """Load GDELT news from JSON file into database"""
    
    # Read the JSON file
    json_path = Path("data/historical/news/real_cocoa_news.json")
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    articles = data.get('articles', [])
    print(f"Found {len(articles)} articles to process")
    
    with Session(engine) as session:
        saved = 0
        duplicates = 0
        
        for article in articles:
            # Check if URL already exists
            url = article.get('url', '')
            if not url:
                continue
                
            existing = session.query(NewsArticle).filter_by(url=url).first()
            if existing:
                duplicates += 1
                continue
            
            # Parse date
            date_str = article.get('date', '')
            try:
                # GDELT format: YYYYMMDDTHHMMSSZ
                if 'T' in date_str:
                    published_date = datetime.strptime(date_str[:15], '%Y%m%dT%H%M%S')
                else:
                    published_date = datetime.now()
            except:
                published_date = datetime.now()
            
            # Create article
            news_article = NewsArticle(
                published_date=published_date,
                fetched_date=datetime.now(),
                url=url,
                title=article.get('headline', '')[:500],
                content=article.get('content', '') or article.get('headline', ''),
                summary=article.get('headline', '')[:500],
                source=article.get('source', 'GDELT'),
                source_type='gdelt_api',
                relevance_score=0.8,
                processed=False
            )
            
            session.add(news_article)
            saved += 1
        
        session.commit()
        print(f"✅ Saved {saved} new articles ({duplicates} duplicates skipped)")
    
    # Check new total
    with Session(engine) as session:
        total = session.query(NewsArticle).count()
        print(f"\n📊 Total news articles in database: {total}")

if __name__ == "__main__":
    print("Loading GDELT news articles...")
    load_gdelt_news()