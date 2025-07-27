#!/usr/bin/env python3
"""
Enhanced Weather Data Collector for Cocoa Growing Regions
Collects comprehensive weather data including soil moisture and evapotranspiration
REAL DATA from multiple sources
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlmodel import Session
from app.core.database import engine
from app.models.weather_data import WeatherData
import numpy as np

class EnhancedWeatherCollector:
    """Collect comprehensive weather data for cocoa regions"""
    
    def __init__(self):
        # Major cocoa growing regions with coordinates
        self.cocoa_regions = {
            'ghana': {
                'kumasi': (6.6666, -1.6163),
                'takoradi': (4.8845, -1.7554),
                'koforidua': (6.0828, -0.2714),
                'ho': (6.6080, 0.4713),
                'sunyani': (7.3398, -2.3266)
            },
            'ivory_coast': {
                'san_pedro': (4.7485, -6.6363),
                'abidjan': (5.3600, -4.0083),
                'yamoussoukro': (6.8276, -5.2893),
                'daloa': (6.8770, -6.4502),
                'gagnoa': (6.1319, -5.9506)
            },
            'nigeria': {
                'ondo': (7.0900, 4.8400),
                'cross_river': (5.8700, 8.5988),
                'akwa_ibom': (5.0077, 7.8537)
            },
            'cameroon': {
                'douala': (4.0511, 9.7679),
                'yaounde': (3.8480, 11.5021),
                'mbam': (4.7333, 11.2500)
            },
            'ecuador': {
                'guayaquil': (-2.1710, -79.9224),
                'machala': (-3.2581, -79.9556),
                'esmeraldas': (0.9681, -79.6517)
            }
        }
        
        # Weather parameters critical for cocoa
        self.weather_params = {
            'temperature': {
                'unit': '°C',
                'optimal_range': (21, 32),
                'critical_low': 15,
                'critical_high': 38
            },
            'precipitation': {
                'unit': 'mm',
                'annual_optimal': (1250, 2500),
                'dry_season_max': 100  # mm per month
            },
            'relative_humidity': {
                'unit': '%',
                'optimal_range': (70, 80),
                'disease_risk_threshold': 85
            },
            'soil_moisture': {
                'unit': 'm³/m³',
                'optimal_range': (0.25, 0.35),
                'stress_threshold': 0.15
            },
            'evapotranspiration': {
                'unit': 'mm',
                'daily_avg': 3.5
            }
        }
    
    def fetch_open_meteo_data(self, lat: float, lon: float, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch comprehensive weather data from Open-Meteo"""
        
        # Historical weather API
        url = "https://archive-api.open-meteo.com/v1/era5"
        
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
                'rain_sum',
                'relative_humidity_2m_mean',
                'dewpoint_2m_mean',
                'windspeed_10m_mean',
                'et0_fao_evapotranspiration',
                'soil_moisture_0_to_10cm_mean',
                'soil_moisture_10_to_40cm_mean',
                'soil_temperature_0_to_7cm_mean'
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
                'rainfall': daily_data['rain_sum'],
                'humidity': daily_data['relative_humidity_2m_mean'],
                'dewpoint': daily_data['dewpoint_2m_mean'],
                'windspeed': daily_data['windspeed_10m_mean'],
                'evapotranspiration': daily_data['et0_fao_evapotranspiration'],
                'soil_moisture_shallow': daily_data['soil_moisture_0_to_10cm_mean'],
                'soil_moisture_deep': daily_data['soil_moisture_10_to_40cm_mean'],
                'soil_temperature': daily_data['soil_temperature_0_to_7cm_mean']
            })
            
            return df
            
        except Exception as e:
            print(f"Error fetching Open-Meteo data for ({lat}, {lon}): {str(e)}")
            return None
    
    def calculate_growing_degree_days(self, df: pd.DataFrame, base_temp: float = 15) -> pd.Series:
        """Calculate Growing Degree Days (GDD) for cocoa"""
        # GDD = (Tmax + Tmin)/2 - Tbase
        # With upper threshold at 35°C
        
        avg_temp = (df['temp_max'] + df['temp_min']) / 2
        avg_temp = avg_temp.clip(base_temp, 35)  # Cap at 35°C
        gdd = avg_temp - base_temp
        gdd = gdd.clip(lower=0)  # Can't be negative
        
        return gdd
    
    def calculate_drought_stress_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate drought stress index based on rainfall and evapotranspiration"""
        # Cumulative water deficit
        water_balance = df['rainfall'] - df['evapotranspiration']
        
        # 30-day rolling deficit
        deficit = water_balance.rolling(30, min_periods=1).sum()
        
        # Normalize to 0-10 scale (0 = no stress, 10 = severe stress)
        stress_index = deficit.clip(upper=0).abs() / 100
        stress_index = stress_index.clip(upper=10)
        
        return stress_index
    
    def calculate_disease_risk_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate black pod disease risk based on temperature and humidity"""
        # Black pod thrives in warm, humid conditions
        # Risk increases when temp 20-30°C and humidity > 80%
        
        temp_risk = ((df['temp_mean'] >= 20) & (df['temp_mean'] <= 30)).astype(float)
        humidity_risk = (df['humidity'] > 80).astype(float)
        
        # Combined risk (0-10 scale)
        risk_index = (temp_risk + humidity_risk) * 5
        
        # Increase risk if consecutive days of high humidity
        consecutive_humid = (df['humidity'] > 85).rolling(7).sum()
        risk_index += (consecutive_humid > 5) * 2
        
        return risk_index.clip(upper=10)
    
    def analyze_weather_anomalies(self, df: pd.DataFrame, location: str) -> Dict:
        """Identify weather anomalies that could impact cocoa production"""
        anomalies = {
            'heat_stress_days': 0,
            'cold_stress_days': 0,
            'drought_periods': 0,
            'flood_risk_days': 0,
            'disease_risk_days': 0
        }
        
        # Heat stress (> 35°C)
        anomalies['heat_stress_days'] = (df['temp_max'] > 35).sum()
        
        # Cold stress (< 15°C)
        anomalies['cold_stress_days'] = (df['temp_min'] < 15).sum()
        
        # Drought periods (30+ days with < 2mm daily rainfall)
        drought_days = (df['rainfall'] < 2).rolling(30).sum()
        anomalies['drought_periods'] = (drought_days >= 28).sum()
        
        # Flood risk (> 100mm in a day)
        anomalies['flood_risk_days'] = (df['rainfall'] > 100).sum()
        
        # Disease risk days
        disease_risk = self.calculate_disease_risk_index(df)
        anomalies['disease_risk_days'] = (disease_risk > 7).sum()
        
        return anomalies
    
    def collect_all_regions(self, days_back: int = 730) -> Dict[str, pd.DataFrame]:
        """Collect weather data for all cocoa regions"""
        end_date = datetime.now().date() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=days_back)
        
        all_data = {}
        
        for country, regions in self.cocoa_regions.items():
            print(f"\nCollecting weather data for {country.upper()}")
            country_data = []
            
            for city, (lat, lon) in regions.items():
                print(f"  Fetching {city} ({lat}, {lon})...")
                
                df = self.fetch_open_meteo_data(
                    lat, lon, 
                    start_date.isoformat(), 
                    end_date.isoformat()
                )
                
                if df is not None:
                    # Add calculated indices
                    df['location'] = city
                    df['country'] = country
                    df['latitude'] = lat
                    df['longitude'] = lon
                    df['gdd'] = self.calculate_growing_degree_days(df)
                    df['drought_stress'] = self.calculate_drought_stress_index(df)
                    df['disease_risk'] = self.calculate_disease_risk_index(df)
                    
                    # Analyze anomalies
                    anomalies = self.analyze_weather_anomalies(df, city)
                    print(f"    Anomalies: {anomalies}")
                    
                    country_data.append(df)
            
            if country_data:
                all_data[country] = pd.concat(country_data, ignore_index=True)
        
        return all_data
    
    def save_to_database(self, weather_data: Dict[str, pd.DataFrame]):
        """Save enhanced weather data to database"""
        with Session(engine) as session:
            total_saved = 0
            
            for country, df in weather_data.items():
                for idx, row in df.iterrows():
                    weather_record = WeatherData(
                        date=row['date'].date(),
                        location=row['location'],
                        country=row['country'],
                        latitude=row['latitude'],
                        longitude=row['longitude'],
                        temperature=row['temp_mean'],
                        temperature_min=row['temp_min'],
                        temperature_max=row['temp_max'],
                        precipitation=row['precipitation'],
                        humidity=row['humidity'],
                        solar_radiation=None,  # Not available in this dataset
                        wind_speed=row['windspeed'],
                        evapotranspiration=row['evapotranspiration'],
                        soil_moisture_0_10cm=row['soil_moisture_shallow'],
                        soil_moisture_10_40cm=row['soil_moisture_deep'],
                        soil_temperature=row['soil_temperature'],
                        growing_degree_days=row['gdd'],
                        drought_index=row['drought_stress'],
                        disease_risk_index=row['disease_risk']
                    )
                    
                    session.add(weather_record)
                    total_saved += 1
                    
                    # Commit in batches
                    if total_saved % 1000 == 0:
                        session.commit()
                        print(f"  Saved {total_saved} records...")
            
            session.commit()
            print(f"\nTotal weather records saved: {total_saved}")
    
    def generate_weather_summary(self, weather_data: Dict[str, pd.DataFrame]) -> Dict:
        """Generate summary statistics for weather conditions"""
        summary = {}
        
        for country, df in weather_data.items():
            # Group by location
            location_stats = {}
            
            for location in df['location'].unique():
                loc_df = df[df['location'] == location]
                
                location_stats[location] = {
                    'avg_temp': loc_df['temp_mean'].mean(),
                    'total_rainfall': loc_df['rainfall'].sum(),
                    'avg_humidity': loc_df['humidity'].mean(),
                    'drought_days': (loc_df['drought_stress'] > 5).sum(),
                    'disease_risk_days': (loc_df['disease_risk'] > 7).sum(),
                    'optimal_days': (
                        (loc_df['temp_mean'].between(21, 32)) & 
                        (loc_df['humidity'].between(70, 80)) &
                        (loc_df['soil_moisture_deep'] > 0.2)
                    ).sum()
                }
            
            summary[country] = location_stats
        
        return summary

def main():
    """Run enhanced weather data collection"""
    collector = EnhancedWeatherCollector()
    
    # Collect data for last 2 years
    print("Starting enhanced weather data collection...")
    weather_data = collector.collect_all_regions(days_back=730)
    
    # Generate summary
    summary = collector.generate_weather_summary(weather_data)
    
    print("\n=== Weather Data Summary ===")
    for country, locations in summary.items():
        print(f"\n{country.upper()}:")
        for location, stats in locations.items():
            print(f"  {location}:")
            print(f"    Average temperature: {stats['avg_temp']:.1f}°C")
            print(f"    Total rainfall: {stats['total_rainfall']:.0f}mm")
            print(f"    Drought stress days: {stats['drought_days']}")
            print(f"    Disease risk days: {stats['disease_risk_days']}")
            print(f"    Optimal growing days: {stats['optimal_days']}")
    
    # Save to database
    if weather_data:
        collector.save_to_database(weather_data)

if __name__ == "__main__":
    main()