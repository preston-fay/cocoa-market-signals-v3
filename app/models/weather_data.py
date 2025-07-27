"""
Weather Data Model - Open-Meteo weather for cocoa regions
"""
from sqlmodel import SQLModel, Field
from datetime import datetime, date as date_type
from typing import Optional

class WeatherData(SQLModel, table=True):
    """Daily weather data from cocoa producing regions"""
    __tablename__ = "weather_data"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Location and time
    date: date_type = Field(index=True)
    location: str = Field(index=True, description="City/region name")
    country: str = Field(description="Country")
    latitude: float = Field()
    longitude: float = Field()
    
    # Temperature metrics (Celsius)
    temp_min: float = Field(description="Minimum temperature")
    temp_max: float = Field(description="Maximum temperature")
    temp_mean: float = Field(description="Mean temperature")
    
    # Precipitation and moisture
    precipitation_mm: float = Field(description="Daily precipitation in mm")
    humidity: float = Field(description="Relative humidity %")
    soil_moisture: Optional[float] = Field(default=None, description="Soil moisture index")
    
    # Risk indicators
    drought_risk: Optional[float] = Field(default=None, description="Drought risk score 0-1")
    flood_risk: Optional[float] = Field(default=None, description="Flood risk score 0-1")
    disease_risk: Optional[float] = Field(default=None, description="Disease risk score 0-1")
    
    # Anomaly detection
    temp_anomaly: Optional[float] = Field(default=None, description="Temperature anomaly vs 30-yr avg")
    precip_anomaly: Optional[float] = Field(default=None, description="Precipitation anomaly vs 30-yr avg")
    
    # Source
    source: str = Field(default="Open-Meteo")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date_type: lambda v: v.isoformat()
        }