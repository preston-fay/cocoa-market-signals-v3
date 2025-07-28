#!/usr/bin/env python3
"""Collect weather data for cocoa regions"""
import sqlite3
import requests
import time
from datetime import datetime

conn = sqlite3.connect("data/cocoa_market_signals_real.db")
cursor = conn.cursor()

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

locations = [
    {"name": "Abidjan", "lat": 5.345, "lon": -4.024, "country": "Côte d'Ivoire"},
    {"name": "Kumasi", "lat": 6.688, "lon": -1.624, "country": "Ghana"},
    {"name": "Lagos", "lat": 6.524, "lon": 3.379, "country": "Nigeria"},
    {"name": "Douala", "lat": 4.051, "lon": 9.768, "country": "Cameroon"},
    {"name": "San Pedro", "lat": 4.748, "lon": -6.627, "country": "Côte d'Ivoire"}
]

total_saved = 0

for loc in locations:
    print(f"Collecting weather for {loc['name']}, {loc['country']}...")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': loc['lat'],
        'longitude': loc['lon'],
        'start_date': '2023-07-01',
        'end_date': '2025-07-25',
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
            
            saved = 0
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
                saved += 1
                
            conn.commit()
            total_saved += saved
            print(f"  ✅ Saved {saved} days")
            time.sleep(1)
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

conn.close()
print(f"\n✅ Total weather records saved: {total_saved}")