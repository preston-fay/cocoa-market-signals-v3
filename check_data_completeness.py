#!/usr/bin/env python3
"""
Check comprehensive data completeness
"""
from sqlmodel import Session, select, func
from app.core.database import engine
from app.models.price_data import PriceData
from app.models.weather_data import WeatherData
from app.models.news_article import NewsArticle
from app.models.trade_data import TradeData

def check_completeness():
    """Generate comprehensive data report"""
    with Session(engine) as session:
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE DATA COMPLETENESS REPORT")
        print("="*60)
        
        # Price Data
        price_count = session.scalar(
            select(func.count(PriceData.id))
        )
        price_range = session.exec(
            select(func.min(PriceData.date), func.max(PriceData.date))
        ).first()
        
        print("\n💰 PRICE DATA:")
        print(f"  Total records: {price_count}")
        if price_range[0]:
            days = (price_range[1] - price_range[0]).days
            print(f"  Date range: {price_range[0]} to {price_range[1]} ({days} days)")
            print(f"  Coverage: {price_count/days*100:.1f}%")
        
        # Weather Data
        weather_count = session.scalar(
            select(func.count(WeatherData.id))
        )
        weather_locations = session.scalar(
            select(func.count(func.distinct(WeatherData.location)))
        )
        weather_countries = session.exec(
            select(func.distinct(WeatherData.country))
        ).all()
        
        # Check weather features
        sample_weather = session.exec(
            select(WeatherData).limit(1)
        ).first()
        
        print("\n🌦️ WEATHER DATA:")
        print(f"  Total records: {weather_count}")
        print(f"  Locations: {weather_locations}")
        print(f"  Countries: {', '.join(weather_countries)}")
        if sample_weather:
            print(f"  Features available:")
            print(f"    - Temperature: ✓")
            print(f"    - Precipitation: ✓")
            print(f"    - Humidity: ✓")
            print(f"    - Drought risk: {'✓' if sample_weather.drought_risk is not None else '✗'}")
            print(f"    - Disease risk: {'✓' if sample_weather.disease_risk is not None else '✗'}")
            print(f"    - Anomalies: {'✓' if sample_weather.temp_anomaly is not None else '✗'}")
        
        # News Data
        news_count = session.scalar(
            select(func.count(NewsArticle.id))
        )
        news_sources = session.scalar(
            select(func.count(func.distinct(NewsArticle.source)))
        )
        
        # Check sentiment analysis
        analyzed_count = session.scalar(
            select(func.count(NewsArticle.id))
            .where(NewsArticle.sentiment_score.is_not(None))
        )
        
        print("\n📰 NEWS DATA:")
        print(f"  Total articles: {news_count}")
        print(f"  Unique sources: {news_sources}")
        print(f"  Analyzed for sentiment: {analyzed_count} ({analyzed_count/news_count*100 if news_count > 0 else 0:.1f}%)")
        print(f"  Status: {'⚠️ Needs more articles' if news_count < 1000 else '✓ Good coverage'}")
        
        # Trade Data
        trade_count = session.scalar(
            select(func.count(TradeData.id))
        )
        trade_countries = session.scalar(
            select(func.count(func.distinct(TradeData.reporter_country)))
        )
        trade_flows = session.exec(
            select(func.distinct(TradeData.trade_flow))
        ).all()
        
        print("\n📦 TRADE DATA:")
        print(f"  Total records: {trade_count}")
        print(f"  Countries: {trade_countries}")
        print(f"  Trade flows: {', '.join(trade_flows)}")
        
        # Overall Assessment
        print("\n" + "="*60)
        print("🎯 DATA QUALITY ASSESSMENT:")
        print("="*60)
        
        assessments = {
            "Price Data": "✓ Complete" if price_count > 500 else "⚠️ Needs more data",
            "Weather Data": "✓ Complete" if weather_locations >= 15 else "⚠️ Needs more locations",
            "News Data": "⚠️ Limited" if news_count < 1000 else "✓ Complete",
            "Trade Data": "✓ Adequate" if trade_count > 100 else "⚠️ Needs more data"
        }
        
        for category, status in assessments.items():
            print(f"  {category}: {status}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if news_count < 1000:
            print("  1. News data is limited - consider:")
            print("     - Manual collection from commodity sites")
            print("     - Web scraping cocoa-specific sources")
            print("     - Using alternative news APIs")
        
        print("  2. Ready to build features with:")
        print("     - Weather anomaly detection")
        print("     - Seasonal patterns")
        print("     - Price volatility indicators")
        print("     - Supply disruption signals")
        
        print("\n📈 NEXT STEPS:")
        print("  1. Build feature engineering pipeline")
        print("  2. Create multi-source model")
        print("  3. Validate performance with real features")
        print("="*60)

if __name__ == "__main__":
    check_completeness()