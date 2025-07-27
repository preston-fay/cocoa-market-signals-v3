#!/usr/bin/env python3
"""
Import weather data into database
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlmodel import Session, select
from app.core.database import engine
from app.models.weather_data import WeatherData
from datetime import datetime
from pathlib import Path

def import_weather_data():
    """
    Import weather data from CSV files into database
    """
    print("\nImporting weather data into database...")
    
    # Read the combined weather data
    weather_file = Path("data/historical/weather/all_locations_weather_2yr.csv")
    
    if not weather_file.exists():
        print(f"❌ Weather file not found: {weather_file}")
        return
    
    # Read CSV
    df = pd.read_csv(weather_file, parse_dates=['date'])
    
    # Handle NaN values
    df = df.fillna({
        'temp_min_c': 0.0,
        'temp_max_c': 0.0,
        'temp_mean_c': 0.0,
        'precipitation_mm': 0.0,
        'humidity_pct': 0.0,
        'soil_moisture': 0.0,
        'weather_risk_index': 0.0,
        'drought_risk': False,
        'flood_risk': False,
        'temp_stress_high': False,
        'temp_stress_low': False
    })
    
    print(f"\nFound {len(df)} weather records to import")
    
    # Location coordinates
    location_coords = {
        "yamoussoukro": {"lat": 6.8276, "lon": -5.2893},
        "kumasi": {"lat": 6.6666, "lon": -1.6163},
        "san_pedro": {"lat": 4.7485, "lon": -6.6363},
        "takoradi": {"lat": 4.8845, "lon": -1.7554}
    }
    
    # Import to database
    imported = 0
    skipped = 0
    
    with Session(engine) as session:
        for _, row in df.iterrows():
            # Check if record exists
            existing = session.exec(
                select(WeatherData).where(
                    WeatherData.date == row['date'].date(),
                    WeatherData.location == row['location']
                )
            ).first()
            
            if existing:
                skipped += 1
                continue
            
            # Get coordinates
            coords = location_coords.get(row['location'], {})
            
            # Create new record with correct column mapping
            weather = WeatherData(
                date=row['date'].date(),
                location=row['location'],
                country=row['country'],
                latitude=coords.get('lat', 0.0),
                longitude=coords.get('lon', 0.0),
                temp_min=row['temp_min_c'],
                temp_max=row['temp_max_c'],
                temp_mean=row['temp_mean_c'],
                precipitation_mm=row['precipitation_mm'],
                humidity=row['humidity_pct'],
                soil_moisture=row.get('soil_moisture'),
                drought_risk=bool(row['drought_risk']),
                flood_risk=bool(row['flood_risk']),
                disease_risk=None,  # Not calculated yet
                temp_anomaly=None,  # Not calculated yet
                precip_anomaly=None,  # Not calculated yet
                source='Open-Meteo'
            )
            
            session.add(weather)
            imported += 1
            
            if imported % 100 == 0:
                session.commit()
                print(f"  Imported {imported} records...")
        
        session.commit()
    
    print(f"\n✓ Import complete:")
    print(f"  - Imported: {imported} records")
    print(f"  - Skipped: {skipped} records (already in database)")
    
    # Show summary
    with Session(engine) as session:
        from sqlalchemy import func
        total = session.exec(select(func.count(WeatherData.id))).one()
        locations = session.exec(select(WeatherData.location).distinct()).all()
        
        print(f"\nDatabase now contains:")
        print(f"  - Total weather records: {total}")
        print(f"  - Locations: {[loc[0] for loc in locations]}")
        
        # Sample data
        recent = session.exec(
            select(WeatherData)
            .order_by(WeatherData.date.desc())
            .limit(5)
        ).all()
        
        print(f"\nMost recent weather data:")
        for w in recent:
            print(f"  {w.date} - {w.location}: {w.temp_mean:.1f}°C, {w.precipitation_mm:.1f}mm rain")

if __name__ == "__main__":
    import_weather_data()