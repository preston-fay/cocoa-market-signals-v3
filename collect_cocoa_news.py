#!/usr/bin/env python3
"""
Simple cocoa news collector using NewsAPI
Requires NEWS_API_KEY environment variable
"""
import requests
from datetime import datetime, timedelta
from sqlmodel import Session
from app.core.database import engine
from app.models.news_article import NewsArticle
import os

def collect_newsapi_articles():
    """Collect cocoa articles from NewsAPI"""
    
    # Get API key
    api_key = os.getenv('NEWS_API_KEY', 'db7ae8c3baec48f2bb5ad3fafdfb7bc5')  # Free tier key
    
    if not api_key:
        print("Please set NEWS_API_KEY environment variable")
        return
    
    # Search parameters
    queries = [
        'cocoa AND (futures OR price OR market)',
        'COCOBOD OR "Ghana cocoa"',
        '"Ivory Coast" AND cocoa',
        'chocolate AND (shortage OR supply)',
        '"black pod" OR "cocoa disease"',
        'cocoa AND weather AND (drought OR rain)'
    ]
    
    all_articles = []
    
    for query in queries:
        print(f"\nSearching: {query}")
        
        # NewsAPI endpoint
        url = 'https://newsapi.org/v2/everything'
        
        # Date range - last 30 days
        to_date = datetime.now()
        from_date = to_date - timedelta(days=30)
        
        params = {
            'q': query,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 100,
            'apiKey': api_key
        }
        
        try:
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                print(f"  Found {len(articles)} articles")
                all_articles.extend(articles)
            else:
                print(f"  Error: {response.status_code}")
                print(f"  {response.text}")
                
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    # Save to database
    if all_articles:
        save_articles(all_articles)

def save_articles(articles):
    """Save articles to database"""
    with Session(engine) as session:
        saved = 0
        duplicates = 0
        
        for article in articles:
            # Check if exists
            url = article.get('url', '')
            if not url:
                continue
                
            existing = session.query(NewsArticle).filter_by(url=url).first()
            if existing:
                duplicates += 1
                continue
            
            # Parse date
            published_str = article.get('publishedAt', '')
            try:
                published_date = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            except:
                published_date = datetime.now()
            
            # Create article
            news_article = NewsArticle(
                published_date=published_date,
                fetched_date=datetime.now(),
                url=url,
                title=article.get('title', '')[:500],
                content=article.get('content', '') or article.get('description', ''),
                summary=article.get('description', '')[:500] if article.get('description') else '',
                source=article.get('source', {}).get('name', 'Unknown'),
                source_type='news_api',
                author=article.get('author', ''),
                relevance_score=0.8,
                processed=False
            )
            
            session.add(news_article)
            saved += 1
        
        session.commit()
        print(f"\n✅ Saved {saved} new articles ({duplicates} duplicates skipped)")

def check_current_status():
    """Check current news data status"""
    with Session(engine) as session:
        total = session.query(NewsArticle).count()
        sources = session.query(NewsArticle.source).distinct().count()
        
        # Date range
        oldest = session.query(NewsArticle.published_date).order_by(NewsArticle.published_date).first()
        newest = session.query(NewsArticle.published_date).order_by(NewsArticle.published_date.desc()).first()
        
        print("\n📊 Current News Data Status:")
        print(f"  Total articles: {total}")
        print(f"  Unique sources: {sources}")
        if oldest and newest:
            print(f"  Date range: {oldest[0].date()} to {newest[0].date()}")

if __name__ == "__main__":
    print("🔍 Collecting Cocoa News Articles...")
    
    # Check current status
    check_current_status()
    
    # Collect new articles
    collect_newsapi_articles()
    
    # Check updated status
    check_current_status()