"""
Signal Model - Store detected market signals from multiple sources
"""
from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional

class Signal(SQLModel, table=True):
    """Market signals detected from various data sources"""
    __tablename__ = "signals"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # When signal was detected
    detected_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    signal_date: datetime = Field(index=True, description="Date the signal applies to")
    
    # Signal details
    signal_type: str = Field(index=True, description="Type: price, volume, weather, technical")
    signal_name: str = Field(description="Specific signal: golden_cross, volume_surge, etc")
    signal_direction: str = Field(description="Direction: bullish, bearish, neutral")
    signal_strength: float = Field(description="Strength score -10 to +10")
    
    # Signal value and description
    signal_value: float = Field(description="Numeric value associated with signal")
    description: str = Field(description="Human-readable signal description")
    
    # Source information
    source: str = Field(description="Data source that generated signal")
    detector: str = Field(description="Algorithm/model that detected signal")
    
    # Validation
    confidence: float = Field(description="Confidence in signal 0-1")
    is_confirmed: bool = Field(default=False, description="Confirmed by multiple sources")
    
    # Outcome tracking
    outcome_measured: bool = Field(default=False)
    outcome_direction: Optional[str] = Field(default=None, description="Actual market move")
    outcome_magnitude: Optional[float] = Field(default=None, description="Size of move %")
    was_accurate: Optional[bool] = Field(default=None)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }