"""
Weather Data Fetcher for Cocoa Growing Regions

Fetches historical and real-time weather data from key cocoa-producing areas.
Focus on Côte d'Ivoire and Ghana which produce ~60% of world's cocoa.
"""

import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
import time


class WeatherDataFetcher:
    """
    Fetches weather data from multiple sources for cocoa regions
    """
    
    def __init__(self):
        self.data_dir = Path("data/historical/weather")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.add("logs/weather_fetcher.log", rotation="1 week")
        
        # Key cocoa-producing locations
        self.locations = {
            "yamoussoukro": {
                "country": "Côte d'Ivoire",
                "lat": 6.8276,
                "lon": -5.2893,
                "importance": "high"
            },
            "kumasi": {
                "country": "Ghana", 
                "lat": 6.6666,
                "lon": -1.6163,
                "importance": "high"
            },
            "san_pedro": {
                "country": "Côte d'Ivoire",
                "lat": 4.7485,
                "lon": -6.6363,
                "importance": "medium"
            },
            "takoradi": {
                "country": "Ghana",
                "lat": 4.8845,
                "lon": -1.7554,
                "importance": "medium"
            }
        }
        
    def fetch_open_meteo_data(self, location: str, start_date: str, 
                             end_date: str) -> pd.DataFrame:
        """
        Fetch historical weather data from Open-Meteo (free, no API key required)
        
        Open-Meteo provides:
        - Temperature (daily min/max/mean)
        - Precipitation
        - Relative humidity
        - Soil moisture
        """
        loc_info = self.locations.get(location)
        if not loc_info:
            logger.error(f"Unknown location: {location}")
            return pd.DataFrame()
            
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            "latitude": loc_info["lat"],
            "longitude": loc_info["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min", 
                "temperature_2m_mean",
                "precipitation_sum",
                "relative_humidity_2m_mean",
                "soil_moisture_0_to_10cm_mean"
            ],
            "timezone": "GMT"
        }
        
        try:
            logger.info(f"Fetching weather data for {location} from Open-Meteo")
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            daily_data = data.get("daily", {})
            df = pd.DataFrame({
                "date": pd.to_datetime(daily_data["time"]),
                "temp_max_c": daily_data["temperature_2m_max"],
                "temp_min_c": daily_data["temperature_2m_min"],
                "temp_mean_c": daily_data["temperature_2m_mean"],
                "precipitation_mm": daily_data["precipitation_sum"],
                "humidity_pct": daily_data["relative_humidity_2m_mean"],
                "soil_moisture": daily_data["soil_moisture_0_to_10cm_mean"]
            })
            
            df.set_index("date", inplace=True)
            df["location"] = location
            df["country"] = loc_info["country"]
            
            logger.info(f"Successfully fetched {len(df)} days of weather data for {location}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Open-Meteo data for {location}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_weather_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weather indicators relevant to cocoa production
        """
        if df.empty:
            return df
            
        # Optimal conditions for cocoa:
        # Temperature: 21-32°C
        # Rainfall: 1250-3000mm annually (roughly 3.4-8.2mm daily average)
        # Humidity: 70-80%
        
        # Temperature stress indicators
        df["temp_stress_high"] = df["temp_max_c"] > 32
        df["temp_stress_low"] = df["temp_min_c"] < 21
        df["temp_optimal"] = (df["temp_mean_c"] >= 21) & (df["temp_mean_c"] <= 32)
        
        # Rainfall indicators
        df["rainfall_7d"] = df["precipitation_mm"].rolling(window=7).sum()
        df["rainfall_30d"] = df["precipitation_mm"].rolling(window=30).sum()
        df["drought_risk"] = df["rainfall_30d"] < 50  # Less than 50mm in 30 days
        df["flood_risk"] = df["rainfall_7d"] > 200     # More than 200mm in 7 days
        
        # Humidity indicators
        df["humidity_optimal"] = (df["humidity_pct"] >= 70) & (df["humidity_pct"] <= 80)
        df["humidity_stress"] = (df["humidity_pct"] < 60) | (df["humidity_pct"] > 90)
        
        # Soil moisture stress
        df["soil_moisture_stress"] = df["soil_moisture"] < 0.2  # Below 20%
        
        # Composite weather risk index (0-1, higher = worse)
        df["weather_risk_index"] = (
            df["temp_stress_high"].astype(int) * 0.2 +
            df["temp_stress_low"].astype(int) * 0.2 +
            df["drought_risk"].astype(int) * 0.25 +
            df["flood_risk"].astype(int) * 0.25 +
            df["humidity_stress"].astype(int) * 0.1
        )
        
        return df
    
    def fetch_all_locations(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch weather data for all cocoa-producing locations
        """
        all_data = {}
        
        for location in self.locations.keys():
            logger.info(f"Fetching data for {location}")
            
            # Fetch data
            df = self.fetch_open_meteo_data(location, start_date, end_date)
            
            if not df.empty:
                # Calculate indicators
                df = self.calculate_weather_indicators(df)
                all_data[location] = df
                
                # Be nice to the API
                time.sleep(1)
            else:
                logger.warning(f"No data retrieved for {location}")
                
        return all_data
    
    def save_weather_data(self, data: Dict[str, pd.DataFrame], 
                         period_name: str = "2yr") -> None:
        """
        Save weather data with metadata
        """
        if not data:
            logger.warning("No weather data to save")
            return
            
        # Combine all locations
        all_dfs = []
        for location, df in data.items():
            all_dfs.append(df)
            
            # Save individual location data
            location_file = self.data_dir / f"{location}_weather_{period_name}.csv"
            df.to_csv(location_file)
            logger.info(f"Saved {location} weather data to {location_file}")
        
        # Combined dataset
        combined_df = pd.concat(all_dfs, axis=0)
        combined_file = self.data_dir / f"all_locations_weather_{period_name}.csv"
        combined_df.to_csv(combined_file)
        
        # Create metadata
        metadata = {
            "source": "Open-Meteo Historical Weather API",
            "locations": list(data.keys()),
            "period": period_name,
            "date_range": {
                "start": combined_df.index.min().strftime("%Y-%m-%d"),
                "end": combined_df.index.max().strftime("%Y-%m-%d")
            },
            "total_records": len(combined_df),
            "metrics": [
                "temperature (min/max/mean)",
                "precipitation",
                "humidity",
                "soil moisture",
                "calculated risk indicators"
            ],
            "last_updated": datetime.now().isoformat()
        }
        
        metadata_file = self.data_dir / f"weather_metadata_{period_name}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved weather metadata to {metadata_file}")
        
        # Create summary statistics
        self._create_weather_summary(data, period_name)
    
    def _create_weather_summary(self, data: Dict[str, pd.DataFrame], 
                               period_name: str) -> None:
        """
        Create summary statistics for weather data
        """
        summary = {
            "period": period_name,
            "locations": {}
        }
        
        for location, df in data.items():
            location_summary = {
                "total_days": len(df),
                "temperature": {
                    "avg_mean": float(df["temp_mean_c"].mean()),
                    "avg_max": float(df["temp_max_c"].mean()),
                    "avg_min": float(df["temp_min_c"].mean()),
                    "extreme_max": float(df["temp_max_c"].max()),
                    "extreme_min": float(df["temp_min_c"].min())
                },
                "rainfall": {
                    "total_mm": float(df["precipitation_mm"].sum()),
                    "daily_avg_mm": float(df["precipitation_mm"].mean()),
                    "max_daily_mm": float(df["precipitation_mm"].max()),
                    "dry_days": int((df["precipitation_mm"] == 0).sum()),
                    "heavy_rain_days": int((df["precipitation_mm"] > 50).sum())
                },
                "risk_indicators": {
                    "temp_stress_days": int(df["temp_stress_high"].sum() + df["temp_stress_low"].sum()),
                    "drought_risk_days": int(df["drought_risk"].sum()),
                    "flood_risk_days": int(df["flood_risk"].sum()),
                    "avg_risk_index": float(df["weather_risk_index"].mean())
                }
            }
            summary["locations"][location] = location_summary
        
        summary_file = self.data_dir / f"weather_summary_{period_name}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Created weather summary at {summary_file}")
    
    def fetch_two_year_weather(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch 2 years of weather data for all locations
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years
        
        logger.info(f"Fetching 2 years of weather data from {start_date} to {end_date}")
        
        data = self.fetch_all_locations(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        if data:
            self.save_weather_data(data, "2yr")
            
        return data


if __name__ == "__main__":
    # Initialize fetcher
    fetcher = WeatherDataFetcher()
    
    # Fetch 2 years of weather data
    weather_data = fetcher.fetch_two_year_weather()
    
    if weather_data:
        print(f"Successfully fetched weather data for {len(weather_data)} locations")
        for location, df in weather_data.items():
            print(f"\n{location}: {len(df)} days of data")
            print(f"Average temperature: {df['temp_mean_c'].mean():.1f}°C")
            print(f"Total rainfall: {df['precipitation_mm'].sum():.0f}mm")
            print(f"Average weather risk: {df['weather_risk_index'].mean():.2f}")
    else:
        print("Failed to fetch weather data")