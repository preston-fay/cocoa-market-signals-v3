#!/usr/bin/env python3
"""
Test script to verify ALL data requirements for the project
Based on CLAUDE.md and initial project specification
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sqlmodel import Session, select
from datetime import datetime, timedelta
from app.core.database import engine
from app.models import PriceData
from pathlib import Path
import json

def test_price_data_requirements():
    """Test 1: Verify we have sufficient price data"""
    print("\n" + "="*60)
    print("TEST 1: PRICE DATA REQUIREMENTS")
    print("="*60)
    
    with Session(engine) as session:
        # Check Yahoo Finance data
        yahoo_prices = session.exec(
            select(PriceData).where(PriceData.source == "Yahoo Finance")
        ).all()
        
        print(f"\n✓ Yahoo Finance data: {len(yahoo_prices)} records")
        if yahoo_prices:
            date_range = f"{min(p.date for p in yahoo_prices)} to {max(p.date for p in yahoo_prices)}"
            print(f"  Date range: {date_range}")
            print(f"  Price range: ${min(p.price for p in yahoo_prices):,.0f} - ${max(p.price for p in yahoo_prices):,.0f}")
        
        # Check ICCO data
        icco_prices = session.exec(
            select(PriceData).where(PriceData.source == "ICCO")
        ).all()
        
        print(f"\n✓ ICCO data: {len(icco_prices)} records")
        
        # Requirements check
        assert len(yahoo_prices) > 365, "Need at least 1 year of daily prices"
        print("\n✅ PASS: Have sufficient price data for analysis")

def test_trade_data_requirements():
    """Test 2: Verify UN Comtrade export data"""
    print("\n" + "="*60)
    print("TEST 2: UN COMTRADE EXPORT DATA REQUIREMENTS")
    print("="*60)
    
    # Check if we have trade data files
    trade_files = list(Path("data/historical/trade").glob("*.json"))
    print(f"\nTrade data files found: {len(trade_files)}")
    for f in trade_files:
        print(f"  - {f.name}")
    
    # Check what's in the metadata
    metadata_file = Path("data/historical/trade/export_data_metadata.json")
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"\nTrade data metadata:")
        print(f"  Period: {metadata.get('period', 'Unknown')}")
        print(f"  Countries: {metadata.get('countries', 'Unknown')}")
    
    # TODO: Need to create TradeData model and import this data
    print("\n❌ MISSING: UN Comtrade data not imported to database")
    print("  Need to:")
    print("  1. Create TradeData model")
    print("  2. Fetch/parse Comtrade export volumes")
    print("  3. Store monthly export data by country")

def test_weather_data_requirements():
    """Test 3: Verify weather data from Open-Meteo"""
    print("\n" + "="*60)
    print("TEST 3: WEATHER DATA REQUIREMENTS")
    print("="*60)
    
    # Check weather data files
    weather_files = list(Path("data/historical/weather").glob("*.json"))
    print(f"\nWeather data files found: {len(weather_files)}")
    for f in weather_files:
        print(f"  - {f.name}")
    
    # Check metadata
    metadata_file = Path("data/historical/weather/weather_metadata_2yr.json")
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"\nWeather data coverage:")
        locations = metadata.get('locations', {})
        for loc, info in locations.items():
            print(f"  {loc}: {info.get('country')} ({info.get('latitude')}, {info.get('longitude')})")
    
    # TODO: Need to create WeatherData model and import
    print("\n❌ MISSING: Weather data not imported to database")
    print("  Need to:")
    print("  1. Create WeatherData model")
    print("  2. Fetch precipitation/temperature for key regions")
    print("  3. Store daily weather metrics")

def test_commoncrawl_requirements():
    """Test 4: Verify CommonCrawl data for news/sentiment"""
    print("\n" + "="*60)
    print("TEST 4: COMMONCRAWL DATA REQUIREMENTS")
    print("="*60)
    
    # Check if we have any CommonCrawl data
    cc_dir = Path("data/commoncrawl")
    if cc_dir.exists():
        cc_files = list(cc_dir.glob("*"))
        print(f"\nCommonCrawl files: {len(cc_files)}")
    else:
        print("\n❌ MISSING: No CommonCrawl data directory")
    
    print("\n❌ MISSING: CommonCrawl integration not implemented")
    print("  Need to:")
    print("  1. Set up CommonCrawl data fetching")
    print("  2. Create NewsArticle or Document model")
    print("  3. Extract cocoa-related news/reports")
    print("  4. Process for sentiment/event detection")
    print("  Note: May need document store (MongoDB/PostgreSQL JSONB)")

def test_all_data_sources():
    """Test 5: Summary of all required data sources"""
    print("\n" + "="*60)
    print("TEST 5: DATA SOURCE SUMMARY")
    print("="*60)
    
    required_sources = {
        "Yahoo Finance": "Daily cocoa futures prices",
        "UN Comtrade": "Monthly export/import volumes",
        "Open-Meteo": "Weather data from production regions",
        "CommonCrawl": "News articles and market reports",
        "ICCO": "Official monthly cocoa prices"
    }
    
    print("\nRequired data sources per CLAUDE.md:")
    for source, description in required_sources.items():
        print(f"  • {source}: {description}")
    
    # Check what we actually have
    with Session(engine) as session:
        sources_in_db = session.exec(
            select(PriceData.source).distinct()
        ).all()
        
    print(f"\nCurrently in database:")
    for source in sources_in_db:
        print(f"  ✓ {source}")
    
    missing = set(required_sources.keys()) - set(sources_in_db)
    if missing:
        print(f"\nMissing sources:")
        for source in missing:
            if source not in ["UN Comtrade", "Open-Meteo", "CommonCrawl"]:
                print(f"  ❌ {source}")

def test_data_freshness():
    """Test 6: Check if data is recent enough"""
    print("\n" + "="*60)
    print("TEST 6: DATA FRESHNESS")
    print("="*60)
    
    with Session(engine) as session:
        # Get most recent data
        latest = session.exec(
            select(PriceData).order_by(PriceData.date.desc()).limit(1)
        ).first()
        
        if latest:
            days_old = (datetime.now().date() - latest.date).days
            print(f"\nMost recent price: {latest.date} (${latest.price:,.0f})")
            print(f"Data is {days_old} days old")
            
            if days_old > 7:
                print("⚠️  WARNING: Data may be stale (>7 days old)")
            else:
                print("✅ Data is reasonably fresh")

if __name__ == "__main__":
    print("="*60)
    print("COCOA MARKET SIGNALS - DATA REQUIREMENTS TEST")
    print("="*60)
    
    test_price_data_requirements()
    test_trade_data_requirements()
    test_weather_data_requirements()
    test_commoncrawl_requirements()
    test_all_data_sources()
    test_data_freshness()
    
    print("\n" + "="*60)
    print("DATA REQUIREMENTS SUMMARY")
    print("="*60)
    print("\n✅ HAVE:")
    print("  - 505 days of Yahoo Finance cocoa futures")
    print("  - 4 months of ICCO official prices")
    print("\n❌ MISSING:")
    print("  - UN Comtrade export/import data")
    print("  - Open-Meteo weather data")
    print("  - CommonCrawl news/sentiment data")
    print("\nNEXT STEPS:")
    print("  1. Create models for TradeData, WeatherData, NewsArticle")
    print("  2. Build data fetchers for each source")
    print("  3. Import all data to database")
    print("  4. Then build REAL predictions on complete data")
    print("="*60)