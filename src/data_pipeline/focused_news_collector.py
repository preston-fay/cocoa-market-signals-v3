#!/usr/bin/env python3
"""
Focused News Collector - Gets cocoa news from accessible sources
Uses RSS feeds and public APIs that don't require authentication
NO FAKE DATA - All sources are real
"""
import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from sqlmodel import Session
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.core.database import engine
from app.models.news_article import NewsArticle
import hashlib

class FocusedNewsCollector:
    """Collect cocoa news from RSS feeds and public sources"""
    
    def __init__(self):
        # RSS feeds that work without authentication
        self.rss_feeds = {
            # Commodity news feeds
            'Investing.com Commodities': {
                'url': 'https://www.investing.com/rss/commodities.rss',
                'filter': ['cocoa', 'chocolate', 'cacao']
            },
            'FXStreet Commodities': {
                'url': 'https://www.fxstreet.com/rss/news/Commodities',
                'filter': ['cocoa', 'chocolate']
            },
            'Nasdaq Commodities': {
                'url': 'https://www.nasdaq.com/feed/rssoutbound',
                'params': 'category=Commodities',
                'filter': ['cocoa', 'futures', 'agricultural']
            },
            'MarketWatch Commodities': {
                'url': 'https://feeds.marketwatch.com/marketwatch/marketpulse/',
                'filter': ['cocoa', 'agricultural', 'commodity']
            },
            # Regional news
            'AllAfrica': {
                'url': 'https://allafrica.com/tools/headlines/rdf/ghana/headlines.rdf',
                'filter': ['cocoa', 'COCOBOD', 'export']
            },
            'Ghana News Agency': {
                'url': 'https://newsghana.com.gh/feed/',
                'filter': ['cocoa', 'COCOBOD', 'farmers']
            },
            # Financial news
            'SeekingAlpha Commodities': {
                'url': 'https://seekingalpha.com/feed/tag/commodities.xml',
                'filter': ['cocoa', 'agricultural']
            },
            'AgriCensus': {
                'url': 'https://www.agricensus.com/Article/RSS-Feed-33.html',
                'filter': ['cocoa', 'west africa']
            }
        }
        
        # Web scraping targets (no auth required)
        self.web_sources = {
            'Cocoa Post': {
                'url': 'https://thecocoapost.com/category/news/',
                'type': 'specialized'
            },
            'Confectionery News': {
                'url': 'https://www.confectionerynews.com/Topic/Cocoa',
                'type': 'specialized'
            },
            'Commodity.com': {
                'url': 'https://commodity.com/agricultural/cocoa/',
                'type': 'analysis'
            }
        }
        
        # Search APIs (using public endpoints)
        self.search_apis = {
            'newsapi': {
                'url': 'https://newsapi.org/v2/everything',
                'params': {
                    'q': 'cocoa OR cacao OR "chocolate futures" OR COCOBOD',
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 100
                },
                'requires_key': True  # We'll skip if no key
            }
        }
    
    async def fetch_rss_feed(self, name: str, config: Dict) -> List[Dict]:
        """Fetch and parse RSS feed"""
        articles = []
        
        try:
            # Parse RSS feed
            feed_url = config['url']
            if 'params' in config:
                feed_url += '?' + config['params']
                
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                print(f"  Warning: Feed parsing issues for {name}")
            
            # Extract articles
            for entry in feed.entries[:50]:  # Limit per feed
                # Check if cocoa-related
                text_to_check = f"{entry.get('title', '')} {entry.get('summary', '')}"
                if not any(keyword in text_to_check.lower() for keyword in config['filter']):
                    continue
                
                # Parse date
                published = datetime.now()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    published = datetime(*entry.updated_parsed[:6])
                
                # Generate ID
                article_id = hashlib.md5(entry.link.encode()).hexdigest()
                
                articles.append({
                    'id': article_id,
                    'title': entry.get('title', ''),
                    'url': entry.get('link', ''),
                    'published_date': published,
                    'content': entry.get('summary', ''),
                    'source': name,
                    'source_type': 'rss_feed'
                })
                
        except Exception as e:
            print(f"  Error with {name}: {str(e)}")
        
        return articles
    
    async def search_news_api(self, query: str, days_back: int = 7) -> List[Dict]:
        """Search using NewsAPI (if key available)"""
        articles = []
        
        # Check for API key in environment
        import os
        api_key = os.getenv('NEWS_API_KEY')
        if not api_key:
            print("  Skipping NewsAPI (no API key)")
            return articles
        
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': query,
                'from': from_date,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 100,
                'apiKey': api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for article in data.get('articles', []):
                            articles.append({
                                'id': hashlib.md5(article['url'].encode()).hexdigest(),
                                'title': article['title'],
                                'url': article['url'],
                                'published_date': datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                                'content': article.get('content', article.get('description', '')),
                                'source': article['source']['name'],
                                'source_type': 'news_api'
                            })
                    else:
                        print(f"  NewsAPI error: {response.status}")
                        
        except Exception as e:
            print(f"  NewsAPI error: {str(e)}")
        
        return articles
    
    async def collect_all_sources(self, days_back: int = 30) -> Dict[str, int]:
        """Collect from all available sources"""
        all_articles = []
        source_counts = {}
        
        print("\n📰 Collecting news from RSS feeds...")
        # RSS feeds (most reliable)
        for name, config in self.rss_feeds.items():
            print(f"  Checking {name}...")
            articles = await self.fetch_rss_feed(name, config)
            all_articles.extend(articles)
            source_counts[name] = len(articles)
            print(f"    Found {len(articles)} cocoa-related articles")
            
            # Rate limiting
            await asyncio.sleep(1)
        
        print("\n🔍 Searching news APIs...")
        # News API search
        api_articles = await self.search_news_api(
            'cocoa OR cacao OR "chocolate futures" OR COCOBOD OR "Ivory Coast" OR Ghana',
            days_back
        )
        all_articles.extend(api_articles)
        source_counts['NewsAPI'] = len(api_articles)
        
        print(f"\n📊 Total articles collected: {len(all_articles)}")
        
        # Save to database
        self.save_to_database(all_articles)
        
        return source_counts
    
    def save_to_database(self, articles: List[Dict]):
        """Save articles to database"""
        with Session(engine) as session:
            saved_count = 0
            duplicate_count = 0
            
            for article_data in articles:
                # Check if already exists
                existing = session.query(NewsArticle).filter_by(url=article_data['url']).first()
                if existing:
                    duplicate_count += 1
                    continue
                
                # Create new article
                article = NewsArticle(
                    published_date=article_data['published_date'],
                    fetched_date=datetime.now(),
                    url=article_data['url'],
                    title=article_data['title'],
                    content=article_data.get('content', ''),
                    summary=article_data.get('content', '')[:500] if article_data.get('content') else '',
                    source=article_data['source'],
                    source_type=article_data['source_type'],
                    relevance_score=0.8,
                    processed=False
                )
                
                session.add(article)
                saved_count += 1
            
            session.commit()
            print(f"\n💾 Saved {saved_count} new articles ({duplicate_count} duplicates skipped)")

async def main():
    """Run focused news collection"""
    collector = FocusedNewsCollector()
    
    # Collect from all sources
    source_counts = await collector.collect_all_sources(days_back=30)
    
    # Report
    print("\n📈 Collection Summary:")
    for source, count in source_counts.items():
        if count > 0:
            print(f"  {source}: {count} articles")

if __name__ == "__main__":
    asyncio.run(main())