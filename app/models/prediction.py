"""
Prediction Model - Store REAL model predictions for tracking accuracy
"""
from sqlmodel import SQLModel, Field
from datetime import datetime, date as date_type
from typing import Optional
import uuid

class Prediction(SQLModel, table=True):
    """Model predictions stored for accuracy tracking"""
    __tablename__ = "predictions"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()), index=True)
    
    # When the prediction was made
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    # What we're predicting
    target_date: date_type = Field(index=True, description="Date being predicted")
    prediction_horizon: int = Field(description="Days ahead predicted (e.g., 7)")
    
    # The prediction
    predicted_price: float = Field(description="Predicted price in USD/tonne")
    confidence_score: float = Field(description="Model confidence 0-1")
    prediction_type: str = Field(description="Type: point, range, probabilistic")
    
    # Prediction range (if applicable)
    predicted_low: Optional[float] = Field(default=None)
    predicted_high: Optional[float] = Field(default=None)
    
    # Model information
    model_name: str = Field(index=True, description="Model that made prediction")
    model_version: Optional[str] = Field(default=None)
    model_role: Optional[str] = Field(default=None, description="Zen role: neutral_analyst, etc")
    
    # Actual outcome (filled in later)
    actual_price: Optional[float] = Field(default=None, description="Actual price on target date")
    error: Optional[float] = Field(default=None, description="Prediction error (actual - predicted)")
    error_percentage: Optional[float] = Field(default=None, description="Error as % of actual")
    
    # Context at prediction time
    current_price: float = Field(description="Price when prediction was made")
    features_used: Optional[str] = Field(default=None, description="JSON of features used")
    market_regime: Optional[str] = Field(default=None, description="Market regime detected")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date_type: lambda v: v.isoformat()
        }