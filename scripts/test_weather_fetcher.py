#!/usr/bin/env python3
"""
Test weather data fetcher
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline.weather_data_fetcher import WeatherDataFetcher
import json
from datetime import datetime, timedelta

def test_weather_api():
    """Test if we can fetch weather data"""
    print("\n" + "="*60)
    print("TESTING WEATHER DATA FETCHER")
    print("="*60)
    
    fetcher = WeatherDataFetcher()
    
    # Test fetching recent data for one location
    print("\nTesting Open-Meteo API...")
    
    try:
        # Try to fetch just 7 days of recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        if hasattr(fetcher, 'fetch_weather_data'):
            print(f"Fetching weather from {start_date.date()} to {end_date.date()}")
            
            # Test with one location
            test_location = list(fetcher.locations.keys())[0]
            print(f"Testing with location: {test_location}")
            
            # Try to fetch data
            data = fetcher.fetch_weather_data(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if data:
                print(f"\n✓ Successfully fetched weather data")
            else:
                print("\n❌ No data returned")
                
        elif hasattr(fetcher, 'fetch_openmeteo_data'):
            # Try direct method
            location_info = fetcher.locations['yamoussoukro']
            url = f"https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude': location_info['lat'],
                'longitude': location_info['lon'],
                'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum',
                'timezone': 'UTC',
                'past_days': 7
            }
            
            import requests
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                print("\n✓ Open-Meteo API is accessible")
                print(f"  Temperature unit: {data.get('daily_units', {}).get('temperature_2m_max')}")
                print(f"  Days returned: {len(data.get('daily', {}).get('time', []))}")
            else:
                print(f"\n❌ API error: {response.status_code}")
                
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nNote: Weather fetcher might need updates for current API")

if __name__ == "__main__":
    test_weather_api()