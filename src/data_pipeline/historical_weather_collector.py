#!/usr/bin/env python3
"""
Historical Weather Collector - Gets historical weather data properly
Uses the correct Open-Meteo endpoints for past data
"""
import requests
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
from sqlmodel import Session
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.core.database import engine
from app.models.weather_data import WeatherData

class HistoricalWeatherCollector:
    """Collect historical weather data for cocoa regions"""
    
    def __init__(self):
        # Extended list of cocoa growing regions
        self.regions = {
            'Ghana': [
                ('Kumasi', 6.6666, -1.6163),
                ('Takoradi', 4.8845, -1.7554),
                ('Koforidua', 6.0828, -0.2714),
                ('Ho', 6.6080, 0.4713),
                ('Sunyani', 7.3398, -2.3266),
                ('Sefwi Wiawso', 6.2077, -2.4856),  # Major cocoa area
                ('Goaso', 6.8004, -2.5225)  # Brong-Ahafo region
            ],
            'Ivory Coast': [
                ('San-Pedro', 4.7485, -6.6363),
                ('Abidjan', 5.3600, -4.0083),
                ('Yamoussoukro', 6.8276, -5.2893),
                ('Daloa', 6.8770, -6.4502),  # Major cocoa hub
                ('Gagnoa', 6.1319, -5.9506),  # Cocoa region
                ('Soubre', 5.7852, -6.6093),  # Southwest cocoa
                ('Duekoue', 6.7417, -7.3422)  # Western cocoa
            ],
            'Nigeria': [
                ('Ondo', 7.0900, 4.8400),
                ('Akure', 7.2571, 5.2058),  # Ondo State capital
                ('Ile-Ife', 7.4905, 4.5521)  # Osun State
            ],
            'Cameroon': [
                ('Douala', 4.0511, 9.7679),
                ('Yaounde', 3.8480, 11.5021),
                ('Mbalmayo', 3.5158, 11.5002),  # Centre region
                ('Ebolowa', 2.9000, 11.1500)  # South region
            ]
        }
    
    def fetch_historical_weather(self, lat: float, lon: float, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch historical weather data"""
        
        # Use the correct historical API endpoint
        url = "https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'daily': [
                'temperature_2m_max',
                'temperature_2m_min',
                'temperature_2m_mean',
                'precipitation_sum',
                'relative_humidity_2m_mean'
            ],
            'timezone': 'GMT'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            daily_data = data['daily']
            df = pd.DataFrame({
                'date': pd.to_datetime(daily_data['time']),
                'temp_max': daily_data['temperature_2m_max'],
                'temp_min': daily_data['temperature_2m_min'],
                'temp_mean': daily_data['temperature_2m_mean'],
                'precipitation': daily_data['precipitation_sum'],
                'humidity': daily_data['relative_humidity_2m_mean']
            })
            
            # Handle None values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            print(f"    Error fetching weather for ({lat}, {lon}): {str(e)}")
            return None
    
    def calculate_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weather indices for cocoa"""
        
        # Drought risk (based on consecutive dry days)
        df['dry_day'] = df['precipitation'] < 1.0  # Less than 1mm is dry
        df['consecutive_dry'] = df['dry_day'].groupby((~df['dry_day']).cumsum()).cumsum()
        df['drought_risk'] = (df['consecutive_dry'] / 30).clip(0, 1)  # Normalize to 0-1
        
        # Flood risk (heavy rainfall)
        df['flood_risk'] = (df['precipitation'] > 50).astype(float) * (df['precipitation'] / 100).clip(0, 1)
        
        # Disease risk (high humidity + warm temps)
        optimal_disease_temp = (df['temp_mean'] >= 20) & (df['temp_mean'] <= 30)
        high_humidity = df['humidity'] > 80
        df['disease_risk'] = (optimal_disease_temp & high_humidity).astype(float)
        
        # Temperature anomaly (simplified - deviation from optimal)
        optimal_temp = 26  # Optimal for cocoa
        df['temp_anomaly'] = (df['temp_mean'] - optimal_temp).abs() / 10  # Normalize
        
        # Precipitation anomaly (deviation from optimal)
        optimal_precip_daily = 4.2  # ~1500mm/year
        df['precip_anomaly'] = (df['precipitation'] - optimal_precip_daily).abs() / 50
        
        return df
    
    def collect_all_regions(self):
        """Collect weather data for all regions"""
        
        # Date range - last 2 years of available data
        end_date = date.today() - timedelta(days=10)  # Archive may have delay
        start_date = end_date - timedelta(days=730)  # 2 years
        
        print(f"\n🌦️ Collecting weather data from {start_date} to {end_date}")
        
        with Session(engine) as session:
            total_saved = 0
            
            for country, locations in self.regions.items():
                print(f"\n📍 {country}:")
                
                for city, lat, lon in locations:
                    print(f"  Fetching {city}...", end='')
                    
                    # Check if we already have recent data
                    existing = session.query(WeatherData).filter(
                        WeatherData.location == city,
                        WeatherData.date >= start_date
                    ).count()
                    
                    if existing > 600:  # Skip if we have substantial data
                        print(f" already have {existing} records")
                        continue
                    
                    # Fetch weather data
                    df = self.fetch_historical_weather(
                        lat, lon,
                        start_date.isoformat(),
                        end_date.isoformat()
                    )
                    
                    if df is None or df.empty:
                        print(" failed")
                        continue
                    
                    # Calculate indices
                    df = self.calculate_indices(df)
                    
                    # Save to database
                    saved = 0
                    for _, row in df.iterrows():
                        weather_record = WeatherData(
                            date=row['date'].date(),
                            location=city,
                            country=country,
                            latitude=lat,
                            longitude=lon,
                            temp_min=float(row['temp_min']),
                            temp_max=float(row['temp_max']),
                            temp_mean=float(row['temp_mean']),
                            precipitation_mm=float(row['precipitation']),
                            humidity=float(row['humidity']),
                            drought_risk=float(row['drought_risk']),
                            flood_risk=float(row['flood_risk']),
                            disease_risk=float(row['disease_risk']),
                            temp_anomaly=float(row['temp_anomaly']),
                            precip_anomaly=float(row['precip_anomaly']),
                            source='Open-Meteo Historical'
                        )
                        
                        session.add(weather_record)
                        saved += 1
                    
                    session.commit()
                    total_saved += saved
                    print(f" saved {saved} records")
            
            print(f"\n✅ Total weather records saved: {total_saved}")

def main():
    """Run historical weather collection"""
    collector = HistoricalWeatherCollector()
    collector.collect_all_regions()

if __name__ == "__main__":
    main()