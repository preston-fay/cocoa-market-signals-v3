#!/usr/bin/env python3
"""
Fetch REAL daily cocoa futures data from Yahoo Finance
NO FAKE DATA - Real market prices only
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sqlmodel import Session, select
from app.core.database import engine
from app.models import PriceData
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_cocoa_futures_data():
    """Fetch real cocoa futures data from Yahoo Finance"""
    
    # Cocoa futures symbols
    symbols = {
        'CC=F': 'Cocoa Futures (NYMEX)',
        'KC=F': 'Coffee C Futures (for comparison)',
    }
    
    # Fetch 2 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    logger.info(f"Fetching cocoa futures from {start_date.date()} to {end_date.date()}")
    
    all_data = []
    
    for symbol, description in symbols.items():
        if symbol != 'CC=F':  # Only cocoa for now
            continue
            
        logger.info(f"Fetching {symbol}: {description}")
        
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                continue
            
            logger.info(f"Retrieved {len(df)} days of data for {symbol}")
            
            # Process each row
            for date, row in df.iterrows():
                price_data = PriceData(
                    date=date.date(),
                    price=row['Close'],
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    volume=row['Volume'],
                    source='Yahoo Finance',
                    contract=symbol,
                    exchange='NYMEX'
                )
                all_data.append(price_data)
                
            # Show summary
            logger.info(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
            logger.info(f"  Price range: ${df['Close'].min():,.0f} to ${df['Close'].max():,.0f}")
            logger.info(f"  Current price: ${df['Close'].iloc[-1]:,.0f}")
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {str(e)}")
            logger.info("Make sure yfinance is installed: pip3 install yfinance")
            return None
    
    return all_data

def update_database_with_yahoo_data():
    """Update database with Yahoo Finance data"""
    
    # First check what we have
    with Session(engine) as session:
        existing_count = len(session.exec(select(PriceData)).all())
        logger.info(f"Database currently has {existing_count} price records")
    
    # Fetch new data
    new_data = fetch_cocoa_futures_data()
    
    if not new_data:
        logger.error("Failed to fetch Yahoo Finance data")
        return
    
    # Add to database
    added = 0
    skipped = 0
    
    for price in new_data:
        with Session(engine) as session:
            # Check if already exists
            existing = session.exec(
                select(PriceData).where(
                    PriceData.date == price.date
                )
            ).first()
            
            if existing:
                skipped += 1
            else:
                session.add(price)
                session.commit()
                added += 1
    
    logger.info(f"Added {added} new records, skipped {skipped} existing")
    
    # Show final status
    with Session(engine) as session:
        total_count = len(session.exec(select(PriceData)).all())
        logger.info(f"Database now has {total_count} total price records")
        
        # Get date range
        prices = session.exec(select(PriceData).order_by(PriceData.date)).all()
        if prices:
            logger.info(f"Date range: {prices[0].date} to {prices[-1].date}")

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("FETCHING REAL COCOA FUTURES DATA")
    logger.info("="*60)
    
    update_database_with_yahoo_data()