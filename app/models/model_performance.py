"""
Model Performance Tracking - REAL accuracy metrics over time
"""
from sqlmodel import SQLModel, Field
from datetime import datetime, date as date_type
from typing import Optional

class ModelPerformance(SQLModel, table=True):
    """Track real model performance metrics over time"""
    __tablename__ = "model_performance"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Time period
    evaluation_date: date_type = Field(index=True)
    period_start: date_type = Field()
    period_end: date_type = Field()
    period_days: int = Field()
    
    # Model identification
    model_name: str = Field(index=True)
    model_version: Optional[str] = Field(default=None)
    model_type: str = Field(description="Type: time_series, ml, ensemble")
    
    # Prediction accuracy metrics
    predictions_made: int = Field(description="Number of predictions in period")
    predictions_evaluated: int = Field(description="Predictions with known outcomes")
    
    # Error metrics (REAL values, not fake)
    mae: float = Field(description="Mean Absolute Error")
    mape: float = Field(description="Mean Absolute Percentage Error")
    rmse: float = Field(description="Root Mean Square Error")
    directional_accuracy: float = Field(description="% of correct direction predictions")
    
    # Financial metrics
    sharpe_ratio: Optional[float] = Field(default=None)
    max_drawdown: Optional[float] = Field(default=None)
    cumulative_return: Optional[float] = Field(default=None)
    
    # Confidence calibration
    avg_confidence: float = Field(description="Average confidence score")
    confidence_calibration: Optional[float] = Field(default=None, description="How well confidence matches accuracy")
    
    # By prediction horizon
    accuracy_1d: Optional[float] = Field(default=None)
    accuracy_7d: Optional[float] = Field(default=None)
    accuracy_30d: Optional[float] = Field(default=None)
    
    # By market regime
    accuracy_low_vol: Optional[float] = Field(default=None)
    accuracy_med_vol: Optional[float] = Field(default=None)
    accuracy_high_vol: Optional[float] = Field(default=None)
    
    # Signal performance
    signals_generated: Optional[int] = Field(default=None)
    signal_accuracy: Optional[float] = Field(default=None)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date_type: lambda v: v.isoformat()
        }