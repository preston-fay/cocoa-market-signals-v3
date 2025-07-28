#!/usr/bin/env python3
"""
Fix weather collection using the ARCHIVE API
"""
import requests
import sqlite3
import time

def collect_weather_properly():
    """Use the archive API that actually works"""
    
    locations = [
        {"name": "Abidjan", "lat": 5.345, "lon": -4.024, "country": "Côte d'Ivoire"},
        {"name": "Kumasi", "lat": 6.688, "lon": -1.624, "country": "Ghana"},
        {"name": "Lagos", "lat": 6.524, "lon": 3.379, "country": "Nigeria"},
        {"name": "Douala", "lat": 4.051, "lon": 9.768, "country": "Cameroon"},
        {"name": "Guayaquil", "lat": -2.170, "lon": -79.922, "country": "Ecuador"}
    ]
    
    conn = sqlite3.connect("data/cocoa_market_signals.db")
    cursor = conn.cursor()
    total_saved = 0
    
    for loc in locations:
        # Use ARCHIVE API for historical data
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': loc['lat'],
            'longitude': loc['lon'],
            'start_date': '2023-01-01',
            'end_date': '2024-12-31',
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum',
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
                
                for i in range(len(dates)):
                    cursor.execute('''
                    INSERT OR REPLACE INTO weather_data
                    (location, date, temp_max, temp_min, temp_mean, rainfall, humidity)
                    VALUES (?, ?, ?, ?, ?, ?, NULL)
                    ''', (
                        f"{loc['name']}, {loc['country']}",
                        dates[i],
                        temp_max[i],
                        temp_min[i],
                        (temp_max[i] + temp_min[i]) / 2 if temp_max[i] and temp_min[i] else None,
                        rainfall[i]
                    ))
                
                conn.commit()
                saved = len(dates)
                total_saved += saved
                print(f"✅ {loc['name']}: {saved} days saved")
                time.sleep(1)  # Be nice to API
                
        except Exception as e:
            print(f"❌ Error with {loc['name']}: {e}")
    
    # VERIFY
    cursor.execute("SELECT COUNT(*) FROM weather_data")
    count = cursor.fetchone()[0]
    conn.close()
    
    print(f"\n🔍 TOTAL WEATHER RECORDS IN DATABASE: {count}")
    return count


if __name__ == "__main__":
    count = collect_weather_properly()
    if count > 0:
        print("✅ Weather data SAVED and VERIFIED!")