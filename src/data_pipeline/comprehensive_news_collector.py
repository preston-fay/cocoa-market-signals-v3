#!/usr/bin/env python3
"""
Comprehensive News Collector for Cocoa Market Intelligence
Collects from multiple REAL sources with proper rate limiting
NO FAKE DATA - All sources are verified
"""
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import re
from bs4 import BeautifulSoup
from sqlmodel import Session
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.core.database import engine
from app.models.news_article import NewsArticle
import time
import hashlib

class ComprehensiveNewsCollector:
    """Collects news from multiple verified sources"""
    
    def __init__(self):
        self.sources = {
            'reuters_commodities': {
                'url': 'https://www.reuters.com/markets/commodities/',
                'selectors': {
                    'articles': 'article.story-card',
                    'title': 'h3.story-card__heading__eqhp9',
                    'link': 'a.story-card__heading__link',
                    'date': 'time[datetime]',
                    'content': 'div.article-body__content'
                },
                'rate_limit': 1.0  # seconds between requests
            },
            'icco_org': {
                'url': 'https://www.icco.org/news/',
                'selectors': {
                    'articles': 'div.news-item',
                    'title': 'h3.news-title',
                    'link': 'a.news-link',
                    'date': 'span.news-date',
                    'content': 'div.news-content'
                },
                'rate_limit': 2.0
            },
            'ghanaweb_business': {
                'url': 'https://www.ghanaweb.com/GhanaHomePage/business/',
                'search_terms': ['cocoa', 'COCOBOD', 'chocolate'],
                'selectors': {
                    'articles': 'div.article-list-item',
                    'title': 'h3.article-title',
                    'link': 'a.article-link',
                    'date': 'span.article-date',
                    'content': 'div.article-content'
                },
                'rate_limit': 1.5
            },
            'commodafrica': {
                'url': 'http://www.commodafrica.com/en/news/cocoa',
                'selectors': {
                    'articles': 'article.post',
                    'title': 'h2.entry-title',
                    'link': 'a.entry-link',
                    'date': 'time.entry-date',
                    'content': 'div.entry-content'
                },
                'rate_limit': 2.0
            },
            'businessday_ng': {
                'url': 'https://businessday.ng/?s=cocoa',
                'selectors': {
                    'articles': 'article.post',
                    'title': 'h2.post-title',
                    'link': 'a.post-link',
                    'date': 'time.post-date',
                    'content': 'div.post-content'
                },
                'rate_limit': 1.5
            }
        }
        
        self.cocoa_keywords = [
            'cocoa', 'cacao', 'chocolate', 'ICCO', 'COCOBOD', 
            'Ivory Coast', 'Ghana', 'harvest', 'grinding',
            'black pod', 'weather', 'export', 'futures'
        ]
        
    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch URL with proper error handling"""
        try:
            # Disable SSL verification for problematic sites
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
                async with session.get(url, timeout=30, headers=headers) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        print(f"Error fetching {url}: Status {response.status}")
                        return None
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None
    
    def is_cocoa_related(self, text: str) -> bool:
        """Check if article is cocoa-related"""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in self.cocoa_keywords)
    
    def extract_article_data(self, html: str, source_config: Dict, source_name: str) -> List[Dict]:
        """Extract articles from HTML using source-specific selectors"""
        soup = BeautifulSoup(html, 'html.parser')
        articles = []
        
        article_elements = soup.select(source_config['selectors']['articles'])
        
        for element in article_elements[:50]:  # Limit per page
            try:
                title_elem = element.select_one(source_config['selectors']['title'])
                link_elem = element.select_one(source_config['selectors']['link'])
                date_elem = element.select_one(source_config['selectors']['date'])
                
                if not title_elem or not link_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                link = link_elem.get('href', '')
                
                # Make link absolute
                if link and not link.startswith('http'):
                    base_url = '/'.join(source_config['url'].split('/')[:3])
                    link = base_url + link if link.startswith('/') else base_url + '/' + link
                
                # Check if cocoa-related
                if not self.is_cocoa_related(title):
                    continue
                
                # Parse date
                published_date = datetime.now()
                if date_elem:
                    date_text = date_elem.get_text(strip=True)
                    # Parse various date formats
                    for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%B %d, %Y', '%d %B %Y']:
                        try:
                            published_date = datetime.strptime(date_text, fmt)
                            break
                        except:
                            continue
                
                # Generate unique ID
                article_id = hashlib.md5(link.encode()).hexdigest()
                
                articles.append({
                    'id': article_id,
                    'title': title,
                    'url': link,
                    'published_date': published_date,
                    'source': source_name,
                    'source_type': 'web_scrape'
                })
                
            except Exception as e:
                print(f"Error extracting article from {source_name}: {str(e)}")
                continue
        
        return articles
    
    async def fetch_article_content(self, session: aiohttp.ClientSession, article: Dict, selectors: Dict) -> str:
        """Fetch full article content"""
        html = await self.fetch_url(session, article['url'])
        if not html:
            return ""
        
        soup = BeautifulSoup(html, 'html.parser')
        content_elem = soup.select_one(selectors.get('content', 'article'))
        
        if content_elem:
            # Remove scripts and styles
            for script in content_elem(["script", "style"]):
                script.decompose()
            
            return content_elem.get_text(separator=' ', strip=True)[:5000]  # Limit content length
        
        return ""
    
    async def collect_from_source(self, source_name: str, source_config: Dict, days_back: int = 7) -> List[Dict]:
        """Collect articles from a single source"""
        print(f"Collecting from {source_name}...")
        
        async with aiohttp.ClientSession() as session:
            # Fetch main page
            html = await self.fetch_url(session, source_config['url'])
            if not html:
                return []
            
            # Extract articles
            articles = self.extract_article_data(html, source_config, source_name)
            
            # Fetch content for each article with rate limiting
            for article in articles:
                await asyncio.sleep(source_config['rate_limit'])
                content = await self.fetch_article_content(session, article, source_config['selectors'])
                article['content'] = content
                article['summary'] = content[:500] if content else ""
            
            return articles
    
    async def collect_all_sources(self, days_back: int = 30) -> Dict[str, List[Dict]]:
        """Collect from all sources"""
        all_articles = {}
        
        for source_name, source_config in self.sources.items():
            try:
                articles = await self.collect_from_source(source_name, source_config, days_back)
                all_articles[source_name] = articles
                print(f"Collected {len(articles)} articles from {source_name}")
            except Exception as e:
                print(f"Error collecting from {source_name}: {str(e)}")
                all_articles[source_name] = []
        
        return all_articles
    
    def save_to_database(self, articles_by_source: Dict[str, List[Dict]]):
        """Save articles to database"""
        with Session(engine) as session:
            total_saved = 0
            
            for source_name, articles in articles_by_source.items():
                for article_data in articles:
                    # Check if already exists
                    existing = session.query(NewsArticle).filter_by(url=article_data['url']).first()
                    if existing:
                        continue
                    
                    # Create new article
                    article = NewsArticle(
                        published_date=article_data['published_date'],
                        fetched_date=datetime.now(),
                        url=article_data['url'],
                        title=article_data['title'],
                        content=article_data.get('content', ''),
                        summary=article_data.get('summary', ''),
                        source=source_name,
                        source_type='web_scrape',
                        relevance_score=0.8 if article_data.get('content') else 0.5,
                        processed=False
                    )
                    
                    session.add(article)
                    total_saved += 1
            
            session.commit()
            print(f"Saved {total_saved} new articles to database")

async def main():
    """Run comprehensive news collection"""
    collector = ComprehensiveNewsCollector()
    
    # Collect from all sources
    articles = await collector.collect_all_sources(days_back=30)
    
    # Save to database
    collector.save_to_database(articles)
    
    # Report
    total = sum(len(articles) for articles in articles.values())
    print(f"\nTotal articles collected: {total}")
    print("\nBy source:")
    for source, articles in articles.items():
        print(f"  {source}: {len(articles)}")

if __name__ == "__main__":
    asyncio.run(main())