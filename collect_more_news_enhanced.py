#!/usr/bin/env python3
"""
Enhanced News Collection - Multiple strategies to get 1000+ articles
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
import urllib.parse

class EnhancedNewsCollector:
    """Collect news from multiple sources with enhanced strategies"""
    
    def __init__(self):
        self.total_collected = 0
        self.session = Session(engine)
        
    def collect_gdelt_enhanced(self):
        """Enhanced GDELT collection with more queries and timeframes"""
        print("\n🔍 Enhanced GDELT Collection...")
        
        # Expanded search queries
        queries = [
            # Direct cocoa terms
            'cocoa', 'cacao', '"cocoa beans"', '"cocoa powder"',
            
            # Price and market
            '"cocoa price"', '"cocoa futures"', '"cocoa market"',
            '"cocoa trading"', '"cocoa commodity"', '"cocoa exchange"',
            
            # Major producers
            '("Ghana" OR "Ivory Coast" OR "Cote d\'Ivoire") AND cocoa',
            '("Nigeria" OR "Cameroon" OR "Ecuador") AND cocoa',
            'COCOBOD', '"Conseil du Café-Cacao"',
            
            # Industry players
            '"Barry Callebaut" cocoa', 'Cargill cocoa', 'Olam cocoa',
            'Nestle cocoa', 'Mars cocoa', 'Hershey cocoa',
            
            # Supply chain
            '"cocoa harvest"', '"cocoa production"', '"cocoa export"',
            '"cocoa shortage"', '"cocoa surplus"', '"cocoa supply"',
            
            # Sustainability
            '"cocoa sustainability"', '"fair trade cocoa"',
            '"cocoa certification"', '"cocoa farming"',
            
            # Weather and disease
            '"cocoa weather"', '"cocoa drought"', '"cocoa disease"',
            '"black pod" cocoa', '"swollen shoot" cocoa',
            
            # Processing
            '"cocoa processing"', '"cocoa grinding"', '"chocolate industry"'
        ]
        
        # Multiple time windows to get historical data
        timespans = ['1week', '1month', '3months', '6months', '1year']
        
        for timespan in timespans:
            print(f"\n  Timespan: {timespan}")
            for query in queries:
                articles = self._fetch_gdelt_articles(query, timespan, max_records=250)
                saved = self._save_articles(articles, f'gdelt_{timespan}')
                if saved > 0:
                    print(f"    Query '{query}': {saved} new articles")
                time.sleep(1)  # Rate limiting
                
    def _fetch_gdelt_articles(self, query: str, timespan: str, max_records: int = 250) -> list:
        """Fetch articles from GDELT API"""
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        params = {
            'query': query,
            'mode': 'artlist',
            'format': 'json',
            'maxrecords': str(max_records),
            'timespan': timespan,
            'sort': 'hybridrel'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get('articles', [])
        except Exception as e:
            print(f"    Error: {str(e)}")
            
        return []
    
    def collect_newsapi_free(self):
        """Collect from NewsAPI free tier (if key available)"""
        print("\n📰 NewsAPI Collection...")
        
        # Check if API key exists in environment
        api_key = os.getenv('NEWSAPI_KEY')
        if not api_key:
            print("  No NewsAPI key found. Set NEWSAPI_KEY environment variable.")
            print("  Get free key at: https://newsapi.org/register")
            return
            
        base_url = "https://newsapi.org/v2/everything"
        
        queries = [
            'cocoa', 'cacao', 'chocolate industry',
            'Ghana cocoa', 'Ivory Coast cocoa',
            'cocoa prices', 'cocoa futures'
        ]
        
        for query in queries:
            params = {
                'q': query,
                'apiKey': api_key,
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 100,
                'from': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            }
            
            try:
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    saved = self._save_newsapi_articles(articles)
                    print(f"  Query '{query}': {saved} new articles")
                else:
                    print(f"  Error: {response.status_code}")
            except Exception as e:
                print(f"  Error: {str(e)}")
                
            time.sleep(1)
    
    def collect_bing_news(self):
        """Collect from Bing News Search (if key available)"""
        print("\n🔍 Bing News Search...")
        
        api_key = os.getenv('BING_API_KEY')
        if not api_key:
            print("  No Bing API key found. Set BING_API_KEY environment variable.")
            print("  Get free key at: https://www.microsoft.com/en-us/bing/apis/bing-news-search-api")
            return
            
        headers = {'Ocp-Apim-Subscription-Key': api_key}
        base_url = "https://api.bing.microsoft.com/v7.0/news/search"
        
        queries = ['cocoa market', 'cocoa prices', 'Ghana cocoa', 'Ivory Coast cocoa']
        
        for query in queries:
            params = {
                'q': query,
                'count': 100,
                'freshness': 'Month',
                'textFormat': 'Raw'
            }
            
            try:
                response = requests.get(base_url, headers=headers, params=params)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('value', [])
                    saved = self._save_bing_articles(articles)
                    print(f"  Query '{query}': {saved} new articles")
            except Exception as e:
                print(f"  Error: {str(e)}")
                
    def collect_rss_feeds(self):
        """Collect from commodity-specific RSS feeds"""
        print("\n📡 RSS Feed Collection...")
        
        feeds = [
            # Commodity specific
            "https://www.confectionerynews.com/rss/topic/cocoa",
            "https://www.foodbusinessnews.net/rss/topic/cocoa",
            "https://www.just-food.com/rss/sector/chocolate-confectionery",
            
            # African business news
            "https://www.businessghana.com/site/rss/news",
            "https://www.africanews.com/rss/africanews/business",
            
            # Commodity news
            "https://www.agriculture.com/rss/news",
            "https://www.agrimoney.com/rss/news",
            "https://www.world-grain.com/rss/topic/commodities"
        ]
        
        import feedparser
        
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                cocoa_entries = []
                
                # Filter for cocoa-related entries
                for entry in feed.entries:
                    text = (entry.get('title', '') + ' ' + 
                           entry.get('summary', '')).lower()
                    
                    if any(term in text for term in ['cocoa', 'cacao', 'chocolate']):
                        cocoa_entries.append(entry)
                
                if cocoa_entries:
                    saved = self._save_rss_entries(cocoa_entries, feed_url)
                    print(f"  {feed_url}: {saved} cocoa articles")
                    
            except Exception as e:
                print(f"  Error with {feed_url}: {str(e)}")
    
    def collect_african_news_sites(self):
        """Direct collection from African news sites"""
        print("\n🌍 African News Sites...")
        
        # Use GDELT with domain-specific searches
        african_domains = [
            'ghanaweb.com', 'myjoyonline.com', 'graphic.com.gh',
            'businessghana.com', 'citinewsroom.com', 'peacefmonline.com',
            'africanews.com', 'allafrica.com', 'theafricareport.com'
        ]
        
        for domain in african_domains:
            query = f'site:{domain} (cocoa OR cacao OR COCOBOD)'
            articles = self._fetch_gdelt_articles(query, '1year', 250)
            saved = self._save_articles(articles, f'african_{domain}')
            if saved > 0:
                print(f"  {domain}: {saved} articles")
            time.sleep(1)
    
    def _save_articles(self, articles: list, source_tag: str) -> int:
        """Save GDELT articles to database"""
        saved = 0
        
        for article in articles:
            url = article.get('url', '').strip()
            if not url:
                continue
                
            # Check if exists
            existing = self.session.exec(
                select(NewsArticle).where(NewsArticle.url == url)
            ).first()
            
            if existing:
                continue
                
            # Parse date
            seendate = article.get('seendate', '')
            try:
                published_date = datetime.strptime(seendate[:15], '%Y%m%dT%H%M%S')
            except:
                published_date = datetime.now()
            
            # Create article
            news_article = NewsArticle(
                published_date=published_date,
                fetched_date=datetime.now(),
                url=url,
                title=article.get('title', '')[:500],
                content=article.get('title', ''),  # GDELT only provides title
                summary=article.get('title', '')[:500],
                source=article.get('domain', 'Unknown'),
                source_type=source_tag,
                relevance_score=0.7,
                processed=False
            )
            
            self.session.add(news_article)
            saved += 1
            
        self.session.commit()
        self.total_collected += saved
        return saved
    
    def _save_newsapi_articles(self, articles: list) -> int:
        """Save NewsAPI articles"""
        saved = 0
        
        for article in articles:
            url = article.get('url', '')
            if not url:
                continue
                
            # Check if exists
            existing = self.session.exec(
                select(NewsArticle).where(NewsArticle.url == url)
            ).first()
            
            if existing:
                continue
            
            # Create article
            news_article = NewsArticle(
                published_date=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                fetched_date=datetime.now(),
                url=url,
                title=article.get('title', '')[:500],
                content=article.get('content', article.get('description', '')),
                summary=article.get('description', '')[:500],
                source=article.get('source', {}).get('name', 'Unknown'),
                source_type='newsapi',
                relevance_score=0.8,
                processed=False
            )
            
            self.session.add(news_article)
            saved += 1
            
        self.session.commit()
        self.total_collected += saved
        return saved
    
    def _save_bing_articles(self, articles: list) -> int:
        """Save Bing News articles"""
        saved = 0
        
        for article in articles:
            url = article.get('url', '')
            if not url:
                continue
                
            # Check if exists
            existing = self.session.exec(
                select(NewsArticle).where(NewsArticle.url == url)
            ).first()
            
            if existing:
                continue
            
            # Create article
            news_article = NewsArticle(
                published_date=datetime.fromisoformat(article['datePublished'].replace('Z', '+00:00')),
                fetched_date=datetime.now(),
                url=url,
                title=article.get('name', '')[:500],
                content=article.get('description', ''),
                summary=article.get('description', '')[:500],
                source=article.get('provider', [{}])[0].get('name', 'Unknown'),
                source_type='bing_news',
                relevance_score=0.8,
                processed=False
            )
            
            self.session.add(news_article)
            saved += 1
            
        self.session.commit()
        self.total_collected += saved
        return saved
    
    def _save_rss_entries(self, entries: list, feed_url: str) -> int:
        """Save RSS feed entries"""
        saved = 0
        
        for entry in entries:
            url = entry.get('link', '')
            if not url:
                continue
                
            # Check if exists
            existing = self.session.exec(
                select(NewsArticle).where(NewsArticle.url == url)
            ).first()
            
            if existing:
                continue
            
            # Parse date
            published = entry.get('published_parsed')
            if published:
                published_date = datetime(*published[:6])
            else:
                published_date = datetime.now()
            
            # Create article
            news_article = NewsArticle(
                published_date=published_date,
                fetched_date=datetime.now(),
                url=url,
                title=entry.get('title', '')[:500],
                content=entry.get('summary', ''),
                summary=entry.get('summary', '')[:500],
                source=feed_url.split('/')[2],  # Domain from URL
                source_type='rss_feed',
                relevance_score=0.7,
                processed=False
            )
            
            self.session.add(news_article)
            saved += 1
            
        self.session.commit()
        self.total_collected += saved
        return saved
    
    def get_collection_stats(self):
        """Get current collection statistics"""
        total = self.session.scalar(select(func.count(NewsArticle.id)))
        sources = self.session.scalar(select(func.count(func.distinct(NewsArticle.source))))
        
        # Date range
        oldest = self.session.scalar(select(func.min(NewsArticle.published_date)))
        newest = self.session.scalar(select(func.max(NewsArticle.published_date)))
        
        return {
            'total_articles': total,
            'unique_sources': sources,
            'date_range': f"{oldest} to {newest}" if oldest else "No articles",
            'new_in_session': self.total_collected
        }
    
    def run_all_collectors(self):
        """Run all available collectors"""
        print("🚀 Enhanced News Collection Starting...")
        print("=" * 60)
        
        # Get initial count
        initial_count = self.session.scalar(select(func.count(NewsArticle.id)))
        print(f"Starting with {initial_count} articles\n")
        
        # Run collectors
        self.collect_gdelt_enhanced()
        self.collect_newsapi_free()
        self.collect_bing_news()
        self.collect_rss_feeds()
        self.collect_african_news_sites()
        
        # Final stats
        stats = self.get_collection_stats()
        print("\n" + "=" * 60)
        print("📊 Collection Complete!")
        print(f"  Total articles: {stats['total_articles']}")
        print(f"  New articles added: {stats['new_in_session']}")
        print(f"  Unique sources: {stats['unique_sources']}")
        print(f"  Date range: {stats['date_range']}")
        
        # Close session
        self.session.close()


def main():
    """Run enhanced news collection"""
    
    collector = EnhancedNewsCollector()
    collector.run_all_collectors()
    
    # Offer to run sentiment analysis
    print("\n💡 Run sentiment analysis on new articles? (y/n)")
    if input().lower() == 'y':
        from src.nlp.sentiment_analysis_engine import CocoaSentimentAnalyzer
        analyzer = CocoaSentimentAnalyzer()
        analyzer.process_all_articles()
        print("✅ Sentiment analysis complete!")


if __name__ == "__main__":
    main()