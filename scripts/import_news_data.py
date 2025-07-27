#!/usr/bin/env python3
"""
Import news data into database
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from sqlmodel import Session, select
from app.core.database import engine
from app.models.news_article import NewsArticle
from datetime import datetime
from pathlib import Path

def import_news_data():
    """
    Import news data from JSON file into database
    """
    print("\nImporting news data into database...")
    
    # Read the news data file
    news_file = Path("data/historical/news/real_cocoa_news.json")
    
    if not news_file.exists():
        print(f"❌ News file not found: {news_file}")
        return
    
    # Read JSON
    with open(news_file, 'r') as f:
        data = json.load(f)
    
    articles = data.get('articles', [])
    print(f"\nFound {len(articles)} news articles to import")
    
    # Import to database
    imported = 0
    skipped = 0
    
    with Session(engine) as session:
        for article in articles:
            # Parse date (GDELT format: YYYYMMDDTHHMMSSZ)
            date_str = article['date']
            try:
                # Remove Z and parse
                date_clean = date_str.rstrip('Z')
                article_date = datetime.strptime(date_clean, '%Y%m%dT%H%M%S')
            except:
                print(f"Skipping article with invalid date: {date_str}")
                skipped += 1
                continue
            
            # Check if article exists (by URL)
            existing = session.exec(
                select(NewsArticle).where(
                    NewsArticle.url == article['url']
                )
            ).first()
            
            if existing:
                skipped += 1
                continue
            
            # Create new article
            news = NewsArticle(
                published_date=article_date,
                title=article['headline'][:500],  # Limit to 500 chars
                url=article['url'],
                source=article['source'],
                source_type='news',
                content=article.get('summary', ''),  # We don't have full content yet
                relevance_score=0.8,  # Default relevance since GDELT found it with cocoa query
                sentiment_score=None,  # Will calculate later
                processed=False
            )
            
            session.add(news)
            imported += 1
            
            if imported % 10 == 0:
                session.commit()
                print(f"  Imported {imported} articles...")
        
        session.commit()
    
    print(f"\n✓ Import complete:")
    print(f"  - Imported: {imported} articles")
    print(f"  - Skipped: {skipped} articles (already in database or invalid)")
    
    # Show summary
    with Session(engine) as session:
        from sqlalchemy import func
        total = session.exec(select(func.count(NewsArticle.id))).one()
        sources = session.exec(select(NewsArticle.source).distinct()).all()
        
        print(f"\nDatabase now contains:")
        print(f"  - Total news articles: {total}")
        print(f"  - Sources: {len(sources)} unique sources")
        
        # Recent articles
        recent = session.exec(
            select(NewsArticle)
            .order_by(NewsArticle.published_date.desc())
            .limit(5)
        ).all()
        
        print(f"\nMost recent articles:")
        for article in recent:
            print(f"  {article.published_date.date()} - {article.title[:80]}...")

if __name__ == "__main__":
    import_news_data()