#!/usr/bin/env python3
"""
Collect HISTORICAL news articles to cover 2023-2024 period
Uses different strategies for historical data
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

class HistoricalNewsCollector:
    """Collect historical news to fill gaps"""
    
    def __init__(self):
        self.session = Session(engine)
        self.articles_collected = 0
        
    def collect_gdelt_historical(self):
        """Use GDELT with specific date ranges for historical data"""
        print("📅 Collecting historical news from GDELT...")
        
        # Define historical periods to search
        periods = [
            # 2023
            ("2023-07-01", "2023-09-30", "Q3 2023"),
            ("2023-10-01", "2023-12-31", "Q4 2023"),
            # 2024
            ("2024-01-01", "2024-03-31", "Q1 2024"),
            ("2024-04-01", "2024-06-30", "Q2 2024"),
            ("2024-07-01", "2024-09-30", "Q3 2024"),
            ("2024-10-01", "2024-12-31", "Q4 2024"),
            # 2025 Q1
            ("2025-01-01", "2025-03-31", "Q1 2025"),
        ]
        
        # Queries focused on major events
        queries = [
            # Basic cocoa terms
            'cocoa', '"cocoa prices"', '"cocoa market"',
            
            # Regional specific
            'Ghana cocoa', 'Ivory Coast cocoa', 'COCOBOD',
            '"Côte d\'Ivoire" cocoa', 'Nigeria cocoa',
            
            # Market events
            '"cocoa futures"', '"cocoa shortage"', '"cocoa harvest"',
            
            # Weather/climate
            '"El Nino" cocoa', 'drought cocoa', 'weather cocoa',
            
            # Major companies
            'Barry Callebaut', 'Cargill cocoa', 'Olam cocoa'
        ]
        
        for start_date, end_date, period_name in periods:
            print(f"\n📆 {period_name} ({start_date} to {end_date})")
            
            for query in queries:
                # Use GDELT's date range parameters
                url = "https://api.gdeltproject.org/api/v2/doc/doc"
                
                params = {
                    'query': f'{query} sourcecountry:ghana OR sourcecountry:ivorycoast',
                    'mode': 'artlist',
                    'format': 'json',
                    'maxrecords': '250',
                    'startdatetime': start_date.replace('-', '') + '000000',
                    'enddatetime': end_date.replace('-', '') + '235959',
                    'sort': 'datedesc'
                }
                
                try:
                    response = requests.get(url, params=params, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        articles = data.get('articles', [])
                        if articles:
                            saved = self._save_articles(articles, f'historical_{period_name}')
                            if saved > 0:
                                print(f"  {query}: {saved} articles")
                except Exception as e:
                    print(f"  Error: {str(e)}")
                
                time.sleep(1)  # Rate limiting
    
    def collect_wayback_machine(self):
        """Use Internet Archive Wayback Machine for historical articles"""
        print("\n🕰️ Checking Wayback Machine archives...")
        
        # Major cocoa news sites to check
        sites = [
            'reuters.com/markets/commodities',
            'bloomberg.com/markets/commodities',
            'confectionerynews.com',
            'cnbc.com/cocoa',
            'ft.com/commodities'
        ]
        
        for site in sites:
            # Query Wayback Machine API
            url = f"http://web.archive.org/cdx/search/cdx"
            params = {
                'url': f'{site}/*cocoa*',
                'output': 'json',
                'from': '202307',
                'to': '202504',
                'limit': '100'
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    results = response.json()
                    if len(results) > 1:  # First row is headers
                        print(f"  {site}: {len(results)-1} snapshots found")
                        # Process snapshots
                        for row in results[1:]:
                            timestamp = row[1]
                            original_url = row[2]
                            status = row[4]
                            
                            if status == '200':
                                # Create article entry
                                self._save_wayback_article(timestamp, original_url, site)
            except Exception as e:
                print(f"  {site}: Error - {str(e)}")
    
    def collect_financial_archives(self):
        """Try financial data providers that might have historical news"""
        print("\n💰 Checking financial news archives...")
        
        # Alpha Vantage News Sentiment (requires free API key)
        av_key = os.getenv('ALPHAVANTAGE_API_KEY')
        if av_key:
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': 'NIB,CC=F',  # Cocoa futures ticker
                'time_from': '20230701T0000',
                'limit': '1000',
                'apikey': av_key
            }
            
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if 'feed' in data:
                        saved = self._save_av_articles(data['feed'])
                        print(f"  Alpha Vantage: {saved} articles")
            except Exception as e:
                print(f"  Alpha Vantage: Error - {str(e)}")
        else:
            print("  No Alpha Vantage API key found")
    
    def collect_specific_events(self):
        """Target specific known events in cocoa market"""
        print("\n🎯 Collecting articles about specific events...")
        
        # Known major events to search for
        events = [
            # 2024 price surge
            ('2024-02-01', '2024-02-29', '"cocoa price surge" OR "cocoa rally" February 2024'),
            ('2024-03-01', '2024-03-31', '"cocoa record high" OR "cocoa $10000" March 2024'),
            
            # Weather events
            ('2023-11-01', '2023-12-31', '"El Nino" "West Africa" cocoa 2023'),
            ('2024-06-01', '2024-07-31', '"cocoa drought" OR "dry weather" Ghana "Ivory Coast" 2024'),
            
            # Harvest reports
            ('2023-10-01', '2023-10-31', '"cocoa harvest" "main crop" Ghana "Ivory Coast" 2023'),
            ('2024-10-01', '2024-10-31', '"cocoa harvest" "main crop" Ghana "Ivory Coast" 2024'),
            
            # Policy changes
            ('2023-10-01', '2023-10-31', '"cocoa price" "farm gate" Ghana COCOBOD 2023'),
            ('2024-09-01', '2024-09-30', '"living income differential" cocoa 2024'),
        ]
        
        for start_date, end_date, query in events:
            print(f"\n  Event: {query}")
            
            url = "https://api.gdeltproject.org/api/v2/doc/doc"
            params = {
                'query': query,
                'mode': 'artlist',
                'format': 'json',
                'maxrecords': '250',
                'startdatetime': start_date.replace('-', '') + '000000',
                'enddatetime': end_date.replace('-', '') + '235959',
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    if articles:
                        saved = self._save_articles(articles, 'event_specific')
                        print(f"    Found {saved} articles")
            except Exception as e:
                print(f"    Error: {str(e)}")
            
            time.sleep(1)
    
    def _save_articles(self, articles: list, source_tag: str) -> int:
        """Save articles to database"""
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
            self.articles_collected += 1
            
        self.session.commit()
        return saved
    
    def _save_wayback_article(self, timestamp: str, url: str, source: str):
        """Save Wayback Machine article reference"""
        # Check if exists
        existing = self.session.exec(
            select(NewsArticle).where(NewsArticle.url == url)
        ).first()
        
        if existing:
            return
        
        # Parse timestamp (YYYYMMDDHHMMSS)
        try:
            published_date = datetime.strptime(timestamp[:14], '%Y%m%d%H%M%S')
        except:
            return
        
        # Create article (limited info from Wayback)
        news_article = NewsArticle(
            published_date=published_date,
            fetched_date=datetime.now(),
            url=url,
            title=f"Archived: {source} cocoa article",
            content="[Archived article - content requires retrieval]",
            summary="Historical article from Wayback Machine",
            source=source,
            source_type='wayback_machine',
            relevance_score=0.5,
            processed=False
        )
        
        self.session.add(news_article)
        self.session.commit()
        self.articles_collected += 1
    
    def _save_av_articles(self, articles: list) -> int:
        """Save Alpha Vantage articles"""
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
            
            # Parse date
            time_published = article.get('time_published', '')
            try:
                # Format: 20240315T123000
                published_date = datetime.strptime(time_published[:15], '%Y%m%dT%H%M%S')
            except:
                continue
            
            # Create article
            news_article = NewsArticle(
                published_date=published_date,
                fetched_date=datetime.now(),
                url=url,
                title=article.get('title', '')[:500],
                content=article.get('summary', ''),
                summary=article.get('summary', '')[:500],
                source=article.get('source', 'Unknown'),
                source_type='alphavantage',
                relevance_score=float(article.get('relevance_score', 0.5)),
                processed=False
            )
            
            self.session.add(news_article)
            saved += 1
            self.articles_collected += 1
            
        self.session.commit()
        return saved
    
    def run_all_collectors(self):
        """Run all historical collectors"""
        print("🏛️ Historical News Collection")
        print("=" * 60)
        
        # Get initial count
        initial_count = self.session.scalar(select(func.count(NewsArticle.id)))
        
        # Get current date coverage
        oldest = self.session.scalar(select(func.min(NewsArticle.published_date)))
        newest = self.session.scalar(select(func.max(NewsArticle.published_date)))
        
        print(f"Current articles: {initial_count}")
        print(f"Current date range: {oldest} to {newest}\n")
        
        # Run collectors
        self.collect_gdelt_historical()
        self.collect_specific_events()
        self.collect_wayback_machine()
        self.collect_financial_archives()
        
        # Final stats
        final_count = self.session.scalar(select(func.count(NewsArticle.id)))
        new_oldest = self.session.scalar(select(func.min(NewsArticle.published_date)))
        new_newest = self.session.scalar(select(func.max(NewsArticle.published_date)))
        
        print("\n" + "=" * 60)
        print("📊 Collection Complete!")
        print(f"Total articles: {final_count} (+{final_count - initial_count})")
        print(f"New date range: {new_oldest} to {new_newest}")
        
        # Show monthly distribution
        monthly_dist = self.session.exec(
            select(
                func.strftime('%Y-%m', NewsArticle.published_date).label('month'),
                func.count(NewsArticle.id).label('count')
            )
            .group_by('month')
            .order_by('month')
        ).all()
        
        print("\n📅 Monthly Distribution:")
        for month, count in monthly_dist[-12:]:  # Last 12 months
            print(f"  {month}: {count} articles")
        
        self.session.close()


def main():
    collector = HistoricalNewsCollector()
    collector.run_all_collectors()


if __name__ == "__main__":
    main()