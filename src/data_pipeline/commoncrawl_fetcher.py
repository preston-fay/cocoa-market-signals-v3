"""
CommonCrawl Data Extraction Agent for Cocoa News Articles
Fetches REAL cocoa-related news from CommonCrawl indexes
"""
import requests
import gzip
import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Optional, Set
from urllib.parse import urlparse
import re
from io import BytesIO
import time
import logging
from pathlib import Path
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommonCrawlCocoaExtractor:
    """Extract cocoa-related news articles from CommonCrawl"""
    
    def __init__(self, db_path: str = "cocoa_market_signals.db"):
        self.db_path = db_path
        self.cc_index_url = "http://index.commoncrawl.org"
        
        # Search terms for cocoa content
        self.search_terms = [
            "cocoa futures",
            "COCOBOD",
            "Ivory Coast cocoa",
            "Ghana cocoa", 
            "cocoa harvest",
            "chocolate shortage",
            "black pod disease",
            "cocoa weather",
            "cocoa price",
            "cocoa production",
            "cocoa export",
            "cocoa beans",
            "Côte d'Ivoire cocoa"
        ]
        
        # Target news domains
        self.target_domains = {
            "reuters.com",
            "bloomberg.com",
            "ft.com",
            "agricensus.com",
            "confectionerynews.com",
            "commodafrica.com",
            "ghanaweb.com",
            "businessghana.com",
            "graphic.com.gh",
            "myjoyonline.com",
            "africanbusinessmagazine.com",
            "theafricareport.com",
            "aljazeera.com",
            "bbc.com",
            "cnbc.com",
            "wsj.com",
            "marketwatch.com",
            "investing.com",
            "tradingeconomics.com"
        }
        
        # Get most recent crawl indexes
        self.crawl_indexes = self._get_recent_crawls()
        
    def _get_recent_crawls(self) -> List[str]:
        """Get list of recent CommonCrawl indexes (2023-2025)"""
        crawls = []
        current_year = datetime.now().year
        
        # Generate crawl IDs for recent months
        for year in range(2023, current_year + 1):
            for month in range(1, 13):
                if year == current_year and month > datetime.now().month:
                    break
                    
                # CommonCrawl naming: CC-MAIN-YYYY-WW where WW is week number
                # Approximate 4 crawls per month
                for week_offset in range(0, 4):
                    week_num = (month - 1) * 4 + week_offset + 1
                    if week_num <= 52:
                        crawl_id = f"CC-MAIN-{year}-{week_num:02d}"
                        crawls.append(crawl_id)
        
        # Return most recent crawls first
        return crawls[-12:]  # Last 12 crawls
        
    def search_index(self, crawl_id: str, search_url: str) -> List[Dict]:
        """Search CommonCrawl index for a specific URL pattern"""
        results = []
        
        try:
            # Query CommonCrawl index
            params = {
                "url": search_url,
                "output": "json",
                "limit": 100
            }
            
            response = requests.get(
                f"{self.cc_index_url}/{crawl_id}-index",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                for line in response.text.strip().split('\n'):
                    if line:
                        results.append(json.loads(line))
                        
        except Exception as e:
            logger.warning(f"Error searching index {crawl_id}: {e}")
            
        return results
        
    def fetch_warc_content(self, warc_record: Dict) -> Optional[str]:
        """Fetch and extract content from WARC file"""
        try:
            # Build S3 URL for WARC file
            filename = warc_record['filename']
            offset = int(warc_record['offset'])
            length = int(warc_record['length'])
            
            # CommonCrawl S3 bucket
            s3_url = f"https://data.commoncrawl.org/{filename}"
            
            # Fetch specific byte range
            headers = {
                'Range': f'bytes={offset}-{offset + length - 1}'
            }
            
            response = requests.get(s3_url, headers=headers, timeout=30)
            
            if response.status_code in [200, 206]:
                # Decompress gzipped content
                content = gzip.decompress(response.content)
                
                # Extract HTML from WARC record
                html_start = content.find(b'<!DOCTYPE')
                if html_start == -1:
                    html_start = content.find(b'<html')
                    
                if html_start != -1:
                    html_content = content[html_start:].decode('utf-8', errors='ignore')
                    return self._extract_article_text(html_content)
                    
        except Exception as e:
            logger.warning(f"Error fetching WARC content: {e}")
            
        return None
        
    def _extract_article_text(self, html: str) -> Optional[str]:
        """Extract article text from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Common article content selectors
            article_selectors = [
                'article', 
                '[role="article"]',
                '.article-content',
                '.story-body',
                '.entry-content',
                'main',
                '.post-content'
            ]
            
            article_text = None
            for selector in article_selectors:
                element = soup.select_one(selector)
                if element:
                    article_text = element.get_text(separator=' ', strip=True)
                    break
                    
            if not article_text:
                # Fallback to body text
                article_text = soup.get_text(separator=' ', strip=True)
                
            # Clean up text
            article_text = re.sub(r'\s+', ' ', article_text)
            article_text = article_text[:10000]  # Limit length
            
            return article_text
            
        except Exception as e:
            logger.warning(f"Error parsing HTML: {e}")
            return None
            
    def is_cocoa_related(self, text: str) -> bool:
        """Check if text is cocoa-related"""
        if not text:
            return False
            
        text_lower = text.lower()
        
        # Must contain at least one search term
        return any(term.lower() in text_lower for term in self.search_terms)
        
    def extract_article_metadata(self, html: str, url: str) -> Dict:
        """Extract article metadata from HTML"""
        metadata = {
            'url': url,
            'title': '',
            'published_date': None,
            'author': None
        }
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title_elem = soup.find('title')
            if not title_elem:
                title_elem = soup.find('h1')
            if title_elem:
                metadata['title'] = title_elem.get_text(strip=True)
                
            # Extract publish date
            date_selectors = [
                ('meta', {'property': 'article:published_time'}),
                ('meta', {'name': 'publish_date'}),
                ('time', {'itemprop': 'datePublished'}),
                ('span', {'class': 'date'})
            ]
            
            for tag, attrs in date_selectors:
                elem = soup.find(tag, attrs)
                if elem:
                    date_str = elem.get('content') or elem.get('datetime') or elem.get_text()
                    metadata['published_date'] = self._parse_date(date_str)
                    if metadata['published_date']:
                        break
                        
            # Extract author
            author_selectors = [
                ('meta', {'name': 'author'}),
                ('span', {'class': 'author'}),
                ('a', {'rel': 'author'})
            ]
            
            for tag, attrs in author_selectors:
                elem = soup.find(tag, attrs)
                if elem:
                    metadata['author'] = elem.get('content') or elem.get_text(strip=True)
                    if metadata['author']:
                        break
                        
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
            
        return metadata
        
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime"""
        if not date_str:
            return None
            
        date_formats = [
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%B %d, %Y',
            '%d %B %Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except:
                continue
                
        return None
        
    def calculate_relevance_score(self, text: str) -> float:
        """Calculate relevance score (0-1) based on content"""
        if not text:
            return 0.0
            
        text_lower = text.lower()
        score = 0.0
        
        # Count occurrences of search terms
        term_counts = sum(text_lower.count(term.lower()) for term in self.search_terms)
        
        # Normalize by text length (per 1000 chars)
        text_length = len(text)
        if text_length > 0:
            score = min(1.0, term_counts / (text_length / 1000))
            
        # Boost score for key terms
        key_terms = ["COCOBOD", "Ghana cocoa", "Ivory Coast cocoa", "cocoa futures"]
        for term in key_terms:
            if term.lower() in text_lower:
                score = min(1.0, score + 0.2)
                
        return round(score, 3)
        
    def save_to_database(self, articles: List[Dict]):
        """Save articles to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                published_date TIMESTAMP,
                fetched_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                url TEXT UNIQUE,
                title TEXT,
                content TEXT,
                summary TEXT,
                source TEXT,
                source_type TEXT DEFAULT 'news',
                author TEXT,
                relevance_score REAL,
                sentiment_score REAL,
                sentiment_label TEXT,
                mentioned_countries TEXT,
                mentioned_companies TEXT,
                topics TEXT,
                market_impact TEXT,
                event_type TEXT,
                processed BOOLEAN DEFAULT 0,
                processing_notes TEXT
            )
        ''')
        
        # Insert articles
        for article in articles:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO news_articles (
                        published_date, url, title, content, source,
                        relevance_score, author, event_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article.get('published_date'),
                    article['url'],
                    article.get('title', ''),
                    article.get('content', ''),
                    article.get('source', ''),
                    article.get('relevance_score', 0.0),
                    article.get('author'),
                    article.get('event_type', 'market')
                ))
            except Exception as e:
                logger.error(f"Error inserting article: {e}")
                
        conn.commit()
        conn.close()
        
    async def fetch_articles_async(self, max_articles: int = 1000):
        """Fetch articles asynchronously for better performance"""
        articles = []
        processed_urls = set()
        
        async with aiohttp.ClientSession() as session:
            for crawl_id in self.crawl_indexes[:3]:  # Process 3 most recent crawls
                logger.info(f"Processing crawl: {crawl_id}")
                
                # Search each target domain
                for domain in self.target_domains:
                    if len(articles) >= max_articles:
                        break
                        
                    search_url = f"*.{domain}/*"
                    index_results = await self._search_index_async(session, crawl_id, search_url)
                    
                    # Process results
                    tasks = []
                    for record in index_results[:20]:  # Limit per domain
                        url = record.get('url', '')
                        
                        # Skip if already processed
                        if url in processed_urls:
                            continue
                            
                        processed_urls.add(url)
                        
                        # Check if URL might be cocoa-related
                        if any(term in url.lower() for term in ['cocoa', 'chocolate', 'ghana', 'ivory']):
                            tasks.append(self._process_record_async(session, record))
                            
                    # Gather results
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        for result in results:
                            if isinstance(result, dict) and result.get('content'):
                                articles.append(result)
                                
        return articles
        
    async def _search_index_async(self, session: aiohttp.ClientSession, 
                                  crawl_id: str, search_url: str) -> List[Dict]:
        """Async version of search_index"""
        results = []
        
        try:
            params = {
                "url": search_url,
                "output": "json",
                "limit": 50
            }
            
            async with session.get(
                f"{self.cc_index_url}/{crawl_id}-index",
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    text = await response.text()
                    for line in text.strip().split('\n'):
                        if line:
                            results.append(json.loads(line))
                            
        except Exception as e:
            logger.warning(f"Error searching index {crawl_id}: {e}")
            
        return results
        
    async def _process_record_async(self, session: aiohttp.ClientSession, 
                                   record: Dict) -> Optional[Dict]:
        """Process a single WARC record asynchronously"""
        content = await self._fetch_warc_content_async(session, record)
        
        if content and self.is_cocoa_related(content):
            url = record.get('url', '')
            domain = urlparse(url).netloc
            
            # Extract metadata
            metadata = {
                'url': url,
                'content': content,
                'source': domain,
                'relevance_score': self.calculate_relevance_score(content),
                'fetched_date': datetime.now(timezone.utc),
                'event_type': self._classify_event_type(content)
            }
            
            return metadata
            
        return None
        
    async def _fetch_warc_content_async(self, session: aiohttp.ClientSession, 
                                       warc_record: Dict) -> Optional[str]:
        """Async version of fetch_warc_content"""
        try:
            filename = warc_record['filename']
            offset = int(warc_record['offset'])
            length = int(warc_record['length'])
            
            s3_url = f"https://data.commoncrawl.org/{filename}"
            
            headers = {
                'Range': f'bytes={offset}-{offset + length - 1}'
            }
            
            async with session.get(
                s3_url, 
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status in [200, 206]:
                    data = await response.read()
                    content = gzip.decompress(data)
                    
                    html_start = content.find(b'<!DOCTYPE')
                    if html_start == -1:
                        html_start = content.find(b'<html')
                        
                    if html_start != -1:
                        html_content = content[html_start:].decode('utf-8', errors='ignore')
                        return self._extract_article_text(html_content)
                        
        except Exception as e:
            logger.warning(f"Error fetching WARC content: {e}")
            
        return None
        
    def _classify_event_type(self, text: str) -> str:
        """Classify the type of event mentioned in the article"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['weather', 'rain', 'drought', 'flood']):
            return 'weather'
        elif any(term in text_lower for term in ['disease', 'pod', 'swollen shoot']):
            return 'disease'
        elif any(term in text_lower for term in ['policy', 'government', 'regulation']):
            return 'policy'
        else:
            return 'market'
            
    def run_extraction(self, max_articles: int = 1000):
        """Main method to run the extraction process"""
        logger.info("Starting CommonCrawl extraction for cocoa news...")
        logger.info(f"Target domains: {len(self.target_domains)}")
        logger.info(f"Search terms: {len(self.search_terms)}")
        logger.info(f"Crawl indexes: {self.crawl_indexes[:3]}")
        
        # Run async extraction
        articles = asyncio.run(self.fetch_articles_async(max_articles))
        
        if articles:
            logger.info(f"Found {len(articles)} cocoa-related articles")
            self.save_to_database(articles)
            logger.info(f"Saved to database: {self.db_path}")
            
            # Print summary
            print("\nExtraction Summary:")
            print(f"Total articles found: {len(articles)}")
            print(f"Average relevance score: {sum(a['relevance_score'] for a in articles) / len(articles):.3f}")
            
            # Show sample articles
            print("\nSample articles:")
            for article in articles[:5]:
                print(f"- {article['url']}")
                print(f"  Relevance: {article['relevance_score']}")
                print(f"  Type: {article['event_type']}")
                print()
        else:
            logger.warning("No articles found")
            
        return articles


def main():
    """Run the CommonCrawl extractor"""
    extractor = CommonCrawlCocoaExtractor()
    extractor.run_extraction(max_articles=500)


if __name__ == "__main__":
    main()