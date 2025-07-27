#!/usr/bin/env python3
"""
Initialize database and load REAL price data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from datetime import datetime, date
from pathlib import Path
from sqlmodel import SQLModel, Session, select
from app.core.database import engine, init_db
from app.models import PriceData
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_icco_prices():
    """Load real ICCO price data from JSON"""
    json_path = Path("data/historical/prices/icco_prices_oct2023_jan2024.json")
    
    if not json_path.exists():
        logger.error(f"ICCO price data not found at {json_path}")
        return []
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    prices = []
    for month, price_data in data['price_data'].items():
        # Parse date
        date_str = price_data.get('date')
        if not date_str:
            continue
            
        price_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        
        # Get price (use average of London/NY if both available)
        if 'london' in price_data and 'new_york' in price_data:
            price = (price_data['london'] + price_data['new_york']) / 2
        elif 'icco_daily_avg' in price_data:
            price = price_data['icco_daily_avg']
        elif 'estimated' in price_data:
            price = price_data['estimated']
        else:
            continue
        
        price_obj = PriceData(
            date=price_date,
            price=price,
            source="ICCO",
            exchange="ICCO Monthly Average"
        )
        prices.append(price_obj)
    
    return prices

def load_daily_price_summary():
    """Load price statistics from summary"""
    json_path = Path("data/historical/prices/daily_price_summary_2yr.json")
    
    if not json_path.exists():
        logger.error(f"Daily price summary not found at {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data

async def initialize_database():
    """Initialize database with real data"""
    logger.info("Creating database tables...")
    
    # Create tables
    SQLModel.metadata.create_all(engine)
    
    # Load ICCO prices
    logger.info("Loading REAL ICCO price data...")
    prices = load_icco_prices()
    
    if prices:
        with Session(engine) as session:
            # Check if data already exists
            existing = session.exec(select(PriceData).limit(1)).first()
            if existing:
                logger.info("Price data already exists in database")
            else:
                # Add all prices
                for price in prices:
                    session.add(price)
                session.commit()
                logger.info(f"Added {len(prices)} REAL price records to database")
    
    # Load summary data
    summary = load_daily_price_summary()
    if summary:
        logger.info("Price statistics from 2-year summary:")
        stats = summary.get('price_statistics', {}).get('cocoa_cc', {})
        logger.info(f"  Min price: ${stats.get('min', 'N/A'):,.0f}")
        logger.info(f"  Max price: ${stats.get('max', 'N/A'):,.0f}")
        logger.info(f"  Mean price: ${stats.get('mean', 'N/A'):,.0f}")
        logger.info(f"  Current price: ${stats.get('current', 'N/A'):,.0f}")
        logger.info(f"  1-year return: {stats.get('1yr_return', 'N/A'):.1%}")
    
    logger.info("Database initialization complete!")
    
    # Show what's in the database
    with Session(engine) as session:
        count = session.exec(select(PriceData)).all()
        logger.info(f"\nDatabase contains {len(count)} price records")
        for record in count:
            logger.info(f"  {record.date}: ${record.price:,.0f} (Source: {record.source})")

if __name__ == "__main__":
    asyncio.run(initialize_database())