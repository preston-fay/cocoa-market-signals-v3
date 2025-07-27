#!/usr/bin/env python3
"""
Quick data collection script that works with existing models
Collects only what's immediately available
"""
import asyncio
import requests
from datetime import datetime, timedelta
from sqlmodel import Session
from app.core.database import engine
from app.models.weather_data import WeatherData
from app.models.news_article import NewsArticle
from src.data_pipeline.comprehensive_news_collector import ComprehensiveNewsCollector

def collect_simple_weather():
    """Collect weather data that fits existing model"""
    print("Collecting weather data for Ghana and Ivory Coast...")
    
    locations = {
        'Ghana': {
            'Kumasi': (6.6666, -1.6163),
            'Takoradi': (4.8845, -1.7554)
        },
        'Ivory Coast': {
            'San-Pedro': (4.7485, -6.6363),
            'Abidjan': (5.3600, -4.0083)
        }
    }
    
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    
    with Session(engine) as session:
        saved_count = 0
        
        for country, cities in locations.items():
            for city, (lat, lon) in cities.items():
                print(f"  Fetching {city}, {country}...")
                
                # Use regular Open-Meteo API (not archive)
                url = "https://api.open-meteo.com/v1/forecast"
                params = {
                    'latitude': lat,
                    'longitude': lon,
                    'past_days': 92,  # Max allowed
                    'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean',
                    'timezone': 'GMT'
                }
                
                try:
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    daily = data['daily']
                    for i in range(len(daily['time'])):
                        date = datetime.strptime(daily['time'][i], '%Y-%m-%d').date()
                        
                        weather_record = WeatherData(
                            date=date,
                            location=city,
                            country=country,
                            latitude=lat,
                            longitude=lon,
                            temp_min=daily['temperature_2m_min'][i],
                            temp_max=daily['temperature_2m_max'][i],
                            temp_mean=(daily['temperature_2m_min'][i] + daily['temperature_2m_max'][i]) / 2,
                            precipitation_mm=daily['precipitation_sum'][i] or 0,
                            humidity=daily['relative_humidity_2m_mean'][i],
                            source='Open-Meteo'
                        )
                        
                        session.add(weather_record)
                        saved_count += 1
                        
                except Exception as e:
                    print(f"    Error: {str(e)}")
                    continue
        
        session.commit()
        print(f"  Saved {saved_count} weather records")

async def collect_simple_news():
    """Collect news from sources that work"""
    print("\nCollecting news articles...")
    
    # Simple news collection from RSS feeds
    rss_feeds = {
        'Investing.com Commodities': 'https://www.investing.com/rss/commodities.rss',
        'FXStreet Commodities': 'https://www.fxstreet.com/rss/news/Commodities'
    }
    
    with Session(engine) as session:
        saved_count = 0
        
        for source, feed_url in rss_feeds.items():
            try:
                import feedparser
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:20]:
                    # Check if cocoa-related
                    if 'cocoa' not in entry.title.lower() and 'cocoa' not in entry.get('summary', '').lower():
                        continue
                    
                    article = NewsArticle(
                        published_date=datetime(*entry.published_parsed[:6]),
                        fetched_date=datetime.now(),
                        url=entry.link,
                        title=entry.title,
                        content=entry.get('summary', ''),
                        summary=entry.get('summary', '')[:500],
                        source=source,
                        source_type='rss_feed',
                        relevance_score=0.7,
                        processed=False
                    )
                    
                    session.add(article)
                    saved_count += 1
                    
            except Exception as e:
                print(f"  Error with {source}: {str(e)}")
                continue
        
        session.commit()
        print(f"  Saved {saved_count} news articles")

def validate_current_data():
    """Check what data we currently have"""
    from sqlmodel import select, func
    
    with Session(engine) as session:
        # Price data
        price_count = session.scalar(
            select(func.count()).select_from(PriceData)
        )
        
        # Weather data
        weather_count = session.scalar(
            select(func.count()).select_from(WeatherData)
        )
        weather_locations = session.scalar(
            select(func.count(func.distinct(WeatherData.location)))
        )
        
        # News data
        news_count = session.scalar(
            select(func.count()).select_from(NewsArticle)
        )
        news_sources = session.scalar(
            select(func.count(func.distinct(NewsArticle.source)))
        )
        
        # Trade data
        trade_count = session.scalar(
            select(func.count()).select_from(TradeData)
        )
        
        print("\n📊 CURRENT DATA STATUS")
        print("=" * 40)
        print(f"Price records: {price_count}")
        print(f"Weather records: {weather_count} ({weather_locations} locations)")
        print(f"News articles: {news_count} ({news_sources} sources)")
        print(f"Trade records: {trade_count}")
        print("=" * 40)

async def main():
    """Run quick data collection"""
    print("Running quick data collection...")
    
    # Collect what we can
    collect_simple_weather()
    await collect_simple_news()
    
    # Show current status
    validate_current_data()

if __name__ == "__main__":
    # Import after checking
    try:
        from app.models.price_data import PriceData
        from app.models.trade_data import TradeData
    except:
        pass
    
    asyncio.run(main())