#!/usr/bin/env python3
"""
COLLECT REAL HISTORICAL DATA WITH CORRECT DATES
NO FAKE DATES - NO BULLSHIT - 100% REAL
"""
import sqlite3
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
import json

class RealHistoricalDataCollector:
    def __init__(self):
        self.db_path = "data/cocoa_market_signals_real.db"
        self.setup_database()
        
    def setup_database(self):
        """Create fresh database with proper schema"""
        print("🔨 Creating new database...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # News table with REAL dates
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            content TEXT,
            published_date DATE NOT NULL,
            source TEXT,
            url TEXT UNIQUE,
            author TEXT,
            sentiment_score REAL,
            relevance_score REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Weather table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT NOT NULL,
            country TEXT NOT NULL,
            date DATE NOT NULL,
            temp_max REAL,
            temp_min REAL,
            temp_mean REAL,
            rainfall REAL,
            humidity REAL,
            wind_speed REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(location, date)
        )''')
        
        # Price table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL UNIQUE,
            open REAL,
            high REAL,
            low REAL,
            close REAL NOT NULL,
            volume INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Model performance tracking
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            train_date DATE NOT NULL,
            test_start DATE NOT NULL,
            test_end DATE NOT NULL,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            feature_importance TEXT,
            parameters TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        conn.commit()
        conn.close()
        print("✅ Database created successfully")
        
    def collect_price_data(self):
        """Get cocoa futures prices from Yahoo Finance"""
        print("\n💰 Collecting cocoa futures prices...")
        
        # Cocoa futures ticker
        ticker = "CC=F"
        
        # Get 2 years of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        try:
            cocoa = yf.Ticker(ticker)
            df = cocoa.history(start=start_date, end=end_date)
            
            if len(df) == 0:
                print("❌ No data returned from Yahoo Finance")
                return 0
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            saved = 0
            for date, row in df.iterrows():
                cursor.execute('''
                INSERT OR REPLACE INTO price_data 
                (date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    date.strftime('%Y-%m-%d'),
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row.get('Volume', 0)
                ))
                saved += cursor.rowcount
                
            conn.commit()
            conn.close()
            
            print(f"✅ Saved {saved} price records")
            print(f"   Date range: {df.index.min()} to {df.index.max()}")
            return saved
            
        except Exception as e:
            print(f"❌ Error collecting prices: {e}")
            return 0
            
    def collect_weather_data(self):
        """Get historical weather data for cocoa regions"""
        print("\n🌤️ Collecting historical weather data...")
        
        locations = [
            {"name": "Abidjan", "lat": 5.345, "lon": -4.024, "country": "Côte d'Ivoire"},
            {"name": "Kumasi", "lat": 6.688, "lon": -1.624, "country": "Ghana"},
            {"name": "Lagos", "lat": 6.524, "lon": 3.379, "country": "Nigeria"},
            {"name": "Douala", "lat": 4.051, "lon": 9.768, "country": "Cameroon"},
            {"name": "San Pedro", "lat": 4.748, "lon": -6.627, "country": "Côte d'Ivoire"}
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        total_saved = 0
        
        # Use Open-Meteo historical weather API
        for loc in locations:
            print(f"  Collecting for {loc['name']}, {loc['country']}...")
            
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                'latitude': loc['lat'],
                'longitude': loc['lon'],
                'start_date': '2022-01-01',
                'end_date': datetime.now().strftime('%Y-%m-%d'),
                'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean,wind_speed_10m_max',
                'timezone': 'auto'
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    
                    dates = data['daily']['time']
                    temp_max = data['daily']['temperature_2m_max']
                    temp_min = data['daily']['temperature_2m_min']
                    rainfall = data['daily']['precipitation_sum']
                    humidity = data['daily'].get('relative_humidity_2m_mean', [None] * len(dates))
                    wind = data['daily'].get('wind_speed_10m_max', [None] * len(dates))
                    
                    for i in range(len(dates)):
                        temp_mean = (temp_max[i] + temp_min[i]) / 2 if temp_max[i] and temp_min[i] else None
                        
                        cursor.execute('''
                        INSERT OR REPLACE INTO weather_data
                        (location, country, date, temp_max, temp_min, temp_mean, rainfall, humidity, wind_speed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            loc['name'],
                            loc['country'],
                            dates[i],
                            temp_max[i],
                            temp_min[i],
                            temp_mean,
                            rainfall[i],
                            humidity[i],
                            wind[i]
                        ))
                        
                    conn.commit()
                    saved = len(dates)
                    total_saved += saved
                    print(f"    ✅ {saved} days saved")
                    time.sleep(1)  # Be nice to API
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
                
        conn.close()
        print(f"✅ Total weather records saved: {total_saved}")
        return total_saved
        
    def collect_news_with_real_dates(self):
        """Collect news from NewsAPI with REAL publication dates"""
        print("\n📰 Collecting news with REAL dates...")
        
        # NewsAPI (free tier allows 100 requests/day)
        # You can get a free key at https://newsapi.org/
        api_key = "YOUR_API_KEY"  # User needs to add their key
        
        # For now, use AlphaVantage news API which includes commodity news
        av_api_key = "demo"  # Using demo key for testing
        
        keywords = ["cocoa", "cacao", "chocolate prices", "cocoa beans", "Ghana cocoa", "Ivory Coast cocoa"]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        total_saved = 0
        
        for keyword in keywords:
            print(f"  Searching for: {keyword}")
            
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': 'COMMODITY:COCOA',
                'topics': keyword,
                'time_from': '20220101T0000',
                'limit': 200,
                'apikey': av_api_key
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'feed' in data:
                        for article in data['feed']:
                            # Extract REAL date
                            pub_date = article.get('time_published', '')[:8]  # YYYYMMDD format
                            if pub_date:
                                date_obj = datetime.strptime(pub_date, '%Y%m%d')
                                
                                cursor.execute('''
                                INSERT OR IGNORE INTO news_articles
                                (title, description, url, published_date, source, relevance_score)
                                VALUES (?, ?, ?, ?, ?, ?)
                                ''', (
                                    article.get('title', ''),
                                    article.get('summary', ''),
                                    article.get('url', ''),
                                    date_obj.strftime('%Y-%m-%d'),
                                    article.get('source', ''),
                                    float(article.get('relevance_score', 0))
                                ))
                                total_saved += cursor.rowcount
                                
                        conn.commit()
                        print(f"    ✅ Found {len(data.get('feed', []))} articles")
                        
            except Exception as e:
                print(f"    ❌ Error: {e}")
                
            time.sleep(12)  # AlphaVantage rate limit: 5 calls/minute
            
        # Also try GDELT with date filters
        print("\n  Searching GDELT for historical articles...")
        base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        # Search month by month to get historical coverage
        current_date = datetime.now()
        for months_back in range(24):  # 2 years
            search_date = current_date - timedelta(days=30*months_back)
            date_str = search_date.strftime('%Y%m%d')
            
            params = {
                'query': 'cocoa (Ghana OR "Ivory Coast" OR "Cote d\'Ivoire")',
                'mode': 'artlist',
                'maxrecords': 100,
                'format': 'json',
                'startdatetime': f'{date_str}000000',
                'enddatetime': f'{date_str}235959'
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'articles' in data:
                        for article in data['articles']:
                            # GDELT provides real dates
                            date_str = article.get('seendate', '')
                            if date_str:
                                try:
                                    date_obj = datetime.strptime(date_str[:8], '%Y%m%d')
                                    
                                    cursor.execute('''
                                    INSERT OR IGNORE INTO news_articles
                                    (title, url, published_date, source)
                                    VALUES (?, ?, ?, ?)
                                    ''', (
                                        article.get('title', ''),
                                        article.get('url', ''),
                                        date_obj.strftime('%Y-%m-%d'),
                                        article.get('domain', '')
                                    ))
                                    total_saved += cursor.rowcount
                                    
                                except:
                                    pass
                                    
                        conn.commit()
                        
            except Exception as e:
                print(f"    ❌ GDELT error for {search_date}: {e}")
                
            time.sleep(1)
            
        conn.close()
        print(f"✅ Total news articles saved: {total_saved}")
        return total_saved
        
    def verify_data_quality(self):
        """Verify we have good data coverage"""
        print("\n🔍 Verifying data quality...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Check price data
        df = pd.read_sql_query("""
            SELECT COUNT(*) as count, 
                   MIN(date) as min_date, 
                   MAX(date) as max_date,
                   COUNT(DISTINCT date) as unique_dates
            FROM price_data
        """, conn)
        print(f"\n💰 Price Data:")
        print(f"   Records: {df['count'].iloc[0]}")
        print(f"   Date range: {df['min_date'].iloc[0]} to {df['max_date'].iloc[0]}")
        print(f"   Unique dates: {df['unique_dates'].iloc[0]}")
        
        # Check weather data
        df = pd.read_sql_query("""
            SELECT COUNT(*) as count,
                   COUNT(DISTINCT location) as locations,
                   MIN(date) as min_date,
                   MAX(date) as max_date
            FROM weather_data
        """, conn)
        print(f"\n🌤️ Weather Data:")
        print(f"   Records: {df['count'].iloc[0]}")
        print(f"   Locations: {df['locations'].iloc[0]}")
        print(f"   Date range: {df['min_date'].iloc[0]} to {df['max_date'].iloc[0]}")
        
        # Check news data
        df = pd.read_sql_query("""
            SELECT COUNT(*) as count,
                   MIN(published_date) as min_date,
                   MAX(published_date) as max_date,
                   COUNT(DISTINCT DATE(published_date)) as unique_dates
            FROM news_articles
        """, conn)
        print(f"\n📰 News Data:")
        print(f"   Articles: {df['count'].iloc[0]}")
        print(f"   Date range: {df['min_date'].iloc[0]} to {df['max_date'].iloc[0]}")
        print(f"   Days with news: {df['unique_dates'].iloc[0]}")
        
        conn.close()
        
    def run_collection(self):
        """Run the complete data collection"""
        print("="*60)
        print("🚀 STARTING REAL DATA COLLECTION")
        print("="*60)
        
        # Collect all data types
        price_count = self.collect_price_data()
        weather_count = self.collect_weather_data()
        news_count = self.collect_news_with_real_dates()
        
        # Verify quality
        self.verify_data_quality()
        
        print("\n" + "="*60)
        print("✅ DATA COLLECTION COMPLETE")
        print("="*60)
        print(f"Prices: {price_count} | Weather: {weather_count} | News: {news_count}")
        print(f"Database: {self.db_path}")
        
        return price_count > 0 and weather_count > 0

if __name__ == "__main__":
    collector = RealHistoricalDataCollector()
    success = collector.run_collection()
    
    if not success:
        print("\n⚠️ WARNING: Some data collection failed!")
        exit(1)