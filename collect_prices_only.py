#!/usr/bin/env python3
"""Quick price collection"""
import yfinance as yf
import sqlite3
from datetime import datetime, timedelta

# Setup database
conn = sqlite3.connect("data/cocoa_market_signals_real.db")
cursor = conn.cursor()

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

# Get cocoa futures
print("Collecting cocoa futures prices...")
ticker = "CC=F"
end_date = datetime.now()
start_date = end_date - timedelta(days=730)

cocoa = yf.Ticker(ticker)
df = cocoa.history(start=start_date, end=end_date)

print(f"Got {len(df)} price records")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Save to database
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

conn.commit()
conn.close()

print(f"✅ Saved {len(df)} price records to database")