#!/usr/bin/env python3
"""
Import trade data into database
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from sqlmodel import Session, select
from app.core.database import engine
from app.models.trade_data import TradeData
from datetime import datetime
from pathlib import Path

def import_trade_data():
    """
    Import trade data from JSON file into database
    """
    print("\nImporting trade data into database...")
    
    # Read the trade data file
    trade_file = Path("data/historical/trade/comtrade_export_data_2023_2024.json")
    
    if not trade_file.exists():
        print(f"❌ Trade file not found: {trade_file}")
        return
    
    # Read JSON
    with open(trade_file, 'r') as f:
        data = json.load(f)
    
    records = data.get('data', [])
    print(f"\nFound {len(records)} trade records to import")
    
    # Import to database
    imported = 0
    skipped = 0
    
    with Session(engine) as session:
        for record in records:
            # Parse period to date (first day of month)
            period_str = record['period']
            year = int(period_str.split('-')[0])
            month = int(period_str.split('-')[1])
            period_date = datetime(year, month, 1).date()
            
            # Check if record exists
            existing = session.exec(
                select(TradeData).where(
                    TradeData.period == period_date,
                    TradeData.reporter_country == record['reporter_country'],
                    TradeData.flow_type == record['flow_type']
                )
            ).first()
            
            if existing:
                skipped += 1
                continue
            
            # Create new record
            trade = TradeData(
                period=period_date,
                reporter_country=record['reporter_country'],
                partner_country='World',  # Aggregate
                flow_type=record['flow_type'],
                commodity_code='1801',  # HS code for cocoa beans
                quantity_tons=record['quantity_tons'],
                trade_value_usd=record['trade_value_usd'],
                unit_value=record['unit_value'],
                source='Generated from documented patterns'
            )
            
            session.add(trade)
            imported += 1
            
            if imported % 50 == 0:
                session.commit()
                print(f"  Imported {imported} records...")
        
        session.commit()
    
    print(f"\n✓ Import complete:")
    print(f"  - Imported: {imported} records")
    print(f"  - Skipped: {skipped} records (already in database)")
    
    # Show summary
    with Session(engine) as session:
        from sqlalchemy import func
        total = session.exec(select(func.count(TradeData.id))).one()
        countries = session.exec(select(TradeData.reporter_country).distinct()).all()
        
        print(f"\nDatabase now contains:")
        print(f"  - Total trade records: {total}")
        print(f"  - Countries: {sorted(countries)}")
        
        # Total volume by country
        print(f"\nTotal export volumes by country:")
        country_totals = session.exec(
            select(
                TradeData.reporter_country,
                func.sum(TradeData.quantity_tons)
            )
            .group_by(TradeData.reporter_country)
            .order_by(func.sum(TradeData.quantity_tons).desc())
        ).all()
        
        for country, volume in country_totals:
            print(f"  - {country}: {volume/1e6:.2f} million tons")

if __name__ == "__main__":
    import_trade_data()