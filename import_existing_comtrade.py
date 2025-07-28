#!/usr/bin/env python3
"""Import existing Comtrade data with REAL dates"""
import sqlite3
import json
import pandas as pd

conn = sqlite3.connect("data/cocoa_market_signals_real.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
CREATE TABLE IF NOT EXISTS comtrade_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reporter_country TEXT NOT NULL,
    partner_country TEXT NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    trade_value_usd REAL,
    quantity_kg REAL,
    unit_price REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(reporter_country, partner_country, year, month)
)''')

print("📊 Importing existing Comtrade data...")

# Load the JSON file
with open('data/historical/trade/comtrade_export_data_2023_2024.json', 'r') as f:
    json_data = json.load(f)

total_saved = 0

# Extract the actual data array
data_records = json_data.get('data', [])

# Process each record
for record in data_records:
    # These should have real dates from Comtrade
    reporter = record.get('reporter_country', '')
    
    # Extract date - period like "2023-01"
    period = str(record.get('period', ''))
    if '-' in period:
        year_month = period.split('-')
        year = int(year_month[0])
        month = int(year_month[1])
        
        # This data has aggregated exports, we need to distribute to partners
        total_value = float(record.get('trade_value_usd', 0))
        total_quantity = float(record.get('quantity_tons', 0)) * 1000  # Convert to kg
        
        # Get destination breakdown
        destinations = record.get('main_destinations', {})
        
        # Save data for each destination
        for partner, share in destinations.items():
            if partner != 'Other':  # Skip 'Other' for now
                partner_value = total_value * share
                partner_quantity = total_quantity * share
                
                cursor.execute('''
                INSERT OR REPLACE INTO comtrade_data
                (reporter_country, partner_country, year, month, trade_value_usd, quantity_kg, unit_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    reporter,
                    partner,
                    year,
                    month,
                    partner_value,
                    partner_quantity,
                    partner_value / partner_quantity if partner_quantity > 0 else 0
                ))
                total_saved += cursor.rowcount
        
        # Also save total export
        cursor.execute('''
        INSERT OR REPLACE INTO comtrade_data
        (reporter_country, partner_country, year, month, trade_value_usd, quantity_kg, unit_price)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            reporter,
            'World',
            year,
            month,
            total_value,
            total_quantity,
            total_value / total_quantity if total_quantity > 0 else 0
        ))
        total_saved += cursor.rowcount

conn.commit()

# Verify what we imported
result = conn.execute("""
    SELECT COUNT(*) as records,
           COUNT(DISTINCT reporter_country) as exporters,
           COUNT(DISTINCT partner_country) as importers,
           MIN(year || '-' || printf('%02d', month)) as earliest,
           MAX(year || '-' || printf('%02d', month)) as latest
    FROM comtrade_data
""").fetchone()

print(f"\n✅ Imported Comtrade data:")
print(f"   Total records: {result[0]}")
print(f"   Exporting countries: {result[1]}")
print(f"   Importing countries: {result[2]}")
print(f"   Date range: {result[3]} to {result[4]}")

# Show sample by major exporters
print("\n📈 Sample monthly exports:")
monthly = conn.execute("""
    SELECT reporter_country,
           year || '-' || printf('%02d', month) as month,
           SUM(quantity_kg) / 1000000 as million_kg,
           SUM(trade_value_usd) / 1000000 as million_usd
    FROM comtrade_data
    WHERE reporter_country IN ('Côte d''Ivoire', 'Ghana', 'Cote d''Ivoire')
    GROUP BY reporter_country, year, month
    ORDER BY year DESC, month DESC
    LIMIT 10
""").fetchall()

for row in monthly:
    print(f"   {row[0]} {row[1]}: {row[2]:.1f}M kg, ${row[3]:.1f}M")

conn.close()
print(f"\n✅ Successfully imported {total_saved} Comtrade records with REAL dates!")