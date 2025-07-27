"""
Price Data Model - REAL historical cocoa prices
"""
from sqlmodel import SQLModel, Field
from datetime import datetime, date as date_type
from typing import Optional
from decimal import Decimal

class PriceData(SQLModel, table=True):
    """Historical cocoa price data from real sources"""
    __tablename__ = "price_data"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    date: date_type = Field(index=True, unique=True)
    
    # Price data
    price: float = Field(description="Daily closing price in USD/tonne")
    open: Optional[float] = Field(default=None)
    high: Optional[float] = Field(default=None)
    low: Optional[float] = Field(default=None)
    volume: Optional[float] = Field(default=None)
    
    # Source tracking
    source: str = Field(description="Data source (e.g., Yahoo Finance, ICCO)")
    contract: Optional[str] = Field(default=None, description="Futures contract (e.g., CC=F)")
    exchange: Optional[str] = Field(default=None, description="Exchange (e.g., NYSE, ICE)")
    
    # Additional metrics
    price_change: Optional[float] = Field(default=None, description="Daily price change %")
    volatility_20d: Optional[float] = Field(default=None, description="20-day rolling volatility")
    ma_20: Optional[float] = Field(default=None, description="20-day moving average")
    ma_50: Optional[float] = Field(default=None, description="50-day moving average")
    rsi: Optional[float] = Field(default=None, description="14-day RSI")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date_type: lambda v: v.isoformat()
        }