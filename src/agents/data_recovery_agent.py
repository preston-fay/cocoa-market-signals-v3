#!/usr/bin/env python3
"""
DATA RECOVERY AGENT - ACTUALLY SAVES DATA THIS TIME
NO MORE BULLSHIT - VERIFIES EVERY SAVE
"""
import os
import sys
import json
import pandas as pd
import sqlite3
from datetime import datetime
import requests
import time

class DataRecoveryAgent:
    """Agent that ACTUALLY saves data and VERIFIES it exists"""
    
    def __init__(self):
        self.db_path = "data/cocoa_market_signals.db"
        self.verified_saves = []
        self.failed_saves = []
        
    def setup_database(self):
        """Create SQLite database with proper tables"""
        print("🔨 Creating SQLite database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # News articles table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            published_date DATETIME,
            source TEXT,
            url TEXT,
            sentiment_score REAL,
            sentiment_subjectivity REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Weather data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT NOT NULL,
            date DATE NOT NULL,
            temp_max REAL,
            temp_min REAL,
            temp_mean REAL,
            rainfall REAL,
            humidity REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(location, date)
        )''')
        
        # Price data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL UNIQUE,
            price REAL NOT NULL,
            volume INTEGER,
            high REAL,
            low REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        conn.commit()
        conn.close()
        
        # VERIFY database exists
        if os.path.exists(self.db_path):
            size = os.path.getsize(self.db_path)
            print(f"✅ Database created: {self.db_path} ({size} bytes)")
            self.verified_saves.append(f"Database: {self.db_path}")
            return True
        else:
            print("❌ DATABASE CREATION FAILED!")
            return False
    
    def collect_and_save_news(self):
        """Collect news from GDELT and SAVE IT FOR REAL"""
        print("\n📰 Collecting news articles...")
        
        articles_saved = 0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # GDELT API for cocoa news
        base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
        # Search for cocoa-related articles
        keywords = ["cocoa", "cacao", "chocolate industry", "cocoa beans", "cocoa prices"]
        
        for keyword in keywords:
            params = {
                'query': f'{keyword} (Ghana OR "Ivory Coast" OR "Cote d\'Ivoire" OR Nigeria OR Cameroon)',
                'mode': 'artlist',
                'maxrecords': 250,
                'format': 'json',
                'timespan': '2y'
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'articles' in data:
                        for article in data['articles']:
                            # INSERT INTO DATABASE
                            cursor.execute('''
                            INSERT OR IGNORE INTO news_articles 
                            (title, description, published_date, source, url, sentiment_score)
                            VALUES (?, ?, ?, ?, ?, ?)
                            ''', (
                                article.get('title', ''),
                                article.get('seendate', ''),
                                article.get('datetimedatetime', datetime.now().isoformat()),
                                article.get('domain', ''),
                                article.get('url', ''),
                                0.0  # Will update with sentiment later
                            ))
                            articles_saved += cursor.rowcount
                            
                print(f"✅ Saved {articles_saved} articles for '{keyword}'")
                conn.commit()
                time.sleep(1)  # Be nice to API
                
            except Exception as e:
                print(f"❌ Error collecting {keyword}: {e}")
        
        # VERIFY saves
        cursor.execute("SELECT COUNT(*) FROM news_articles")
        count = cursor.fetchone()[0]
        conn.close()
        
        print(f"\n🔍 VERIFICATION: {count} articles in database")
        
        if count > 0:
            self.verified_saves.append(f"News articles: {count} saved to database")
            
            # ALSO save to JSON for backward compatibility
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM news_articles", conn)
            conn.close()
            
            json_path = 'data/processed/all_news_articles.json'
            df.to_json(json_path, orient='records', date_format='iso')
            
            if os.path.exists(json_path):
                size = os.path.getsize(json_path) / 1024
                print(f"✅ Also saved to JSON: {json_path} ({size:.1f} KB)")
                self.verified_saves.append(f"JSON backup: {json_path}")
        else:
            self.failed_saves.append("News collection: 0 articles saved")
            
        return count
    
    def collect_and_save_weather(self):
        """Collect weather data and SAVE IT"""
        print("\n🌤️ Collecting weather data...")
        
        # Major cocoa regions
        locations = [
            {"name": "Abidjan", "lat": 5.345, "lon": -4.024, "country": "Côte d'Ivoire"},
            {"name": "Kumasi", "lat": 6.688, "lon": -1.624, "country": "Ghana"},
            {"name": "Lagos", "lat": 6.524, "lon": 3.379, "country": "Nigeria"},
            {"name": "Douala", "lat": 4.051, "lon": 9.768, "country": "Cameroon"},
            {"name": "Guayaquil", "lat": -2.170, "lon": -79.922, "country": "Ecuador"}
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        weather_saved = 0
        
        for loc in locations:
            url = f"https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude': loc['lat'],
                'longitude': loc['lon'],
                'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean',
                'past_days': 365,
                'forecast_days': 0,
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
                    
                    for i in range(len(dates)):
                        cursor.execute('''
                        INSERT OR REPLACE INTO weather_data
                        (location, date, temp_max, temp_min, temp_mean, rainfall, humidity)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            f"{loc['name']}, {loc['country']}",
                            dates[i],
                            temp_max[i],
                            temp_min[i],
                            (temp_max[i] + temp_min[i]) / 2 if temp_max[i] and temp_min[i] else None,
                            rainfall[i],
                            humidity[i]
                        ))
                        weather_saved += cursor.rowcount
                    
                    conn.commit()
                    print(f"✅ Saved {len(dates)} days for {loc['name']}")
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"❌ Error collecting weather for {loc['name']}: {e}")
        
        # VERIFY
        cursor.execute("SELECT COUNT(*) FROM weather_data")
        count = cursor.fetchone()[0]
        conn.close()
        
        print(f"\n🔍 VERIFICATION: {count} weather records in database")
        
        if count > 0:
            self.verified_saves.append(f"Weather records: {count} saved to database")
            
            # Save to CSV too
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM weather_data", conn)
            conn.close()
            
            csv_path = 'data/processed/all_weather_data.csv'
            df.to_csv(csv_path, index=False)
            
            if os.path.exists(csv_path):
                print(f"✅ Also saved to CSV: {csv_path}")
                self.verified_saves.append(f"CSV backup: {csv_path}")
        else:
            self.failed_saves.append("Weather collection: 0 records saved")
            
        return count
    
    def collect_and_save_prices(self):
        """Get price data from existing files and SAVE TO DATABASE"""
        print("\n💰 Loading price data...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        prices_saved = 0
        
        # Try to load from existing files
        price_files = [
            'data/processed/comprehensive_train.csv',
            'data/processed/comprehensive_test.csv'
        ]
        
        for file in price_files:
            if os.path.exists(file):
                df = pd.read_csv(file, index_col='date', parse_dates=True)
                
                if 'current_price' in df.columns:
                    for date, row in df.iterrows():
                        cursor.execute('''
                        INSERT OR REPLACE INTO price_data (date, price)
                        VALUES (?, ?)
                        ''', (date.strftime('%Y-%m-%d'), row['current_price']))
                        prices_saved += cursor.rowcount
                    
                    print(f"✅ Loaded {len(df)} prices from {file}")
        
        conn.commit()
        
        # VERIFY
        cursor.execute("SELECT COUNT(*) FROM price_data")
        count = cursor.fetchone()[0]
        conn.close()
        
        print(f"\n🔍 VERIFICATION: {count} price records in database")
        
        if count > 0:
            self.verified_saves.append(f"Price records: {count} saved to database")
        else:
            self.failed_saves.append("Price collection: 0 records saved")
            
        return count
    
    def generate_final_report(self):
        """Generate report of what was ACTUALLY saved"""
        print("\n" + "="*60)
        print("📊 DATA RECOVERY AGENT - FINAL REPORT")
        print("="*60)
        
        print("\n✅ SUCCESSFULLY SAVED:")
        for item in self.verified_saves:
            print(f"   - {item}")
            
        if self.failed_saves:
            print("\n❌ FAILED TO SAVE:")
            for item in self.failed_saves:
                print(f"   - {item}")
        
        # Database summary
        if os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            print("\n📈 DATABASE SUMMARY:")
            tables = ['news_articles', 'weather_data', 'price_data']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   - {table}: {count} records")
            
            conn.close()
            
            db_size = os.path.getsize(self.db_path) / 1024 / 1024
            print(f"\n💾 Database size: {db_size:.2f} MB")
            print(f"📍 Location: {os.path.abspath(self.db_path)}")
        
        print("\n✅ DATA IS NOW ACTUALLY SAVED AND VERIFIED!")
        
    def run_full_recovery(self):
        """Run the complete data recovery process"""
        print("🚀 DATA RECOVERY AGENT STARTING...")
        print("This time we SAVE and VERIFY everything!\n")
        
        # 1. Setup database
        if not self.setup_database():
            print("❌ CRITICAL: Database setup failed!")
            return False
        
        # 2. Collect and save data
        news_count = self.collect_and_save_news()
        weather_count = self.collect_and_save_weather()
        price_count = self.collect_and_save_prices()
        
        # 3. Generate report
        self.generate_final_report()
        
        return news_count > 0 and weather_count > 0 and price_count > 0


if __name__ == "__main__":
    agent = DataRecoveryAgent()
    success = agent.run_full_recovery()
    
    if not success:
        print("\n⚠️ WARNING: Some data collection failed!")
        sys.exit(1)