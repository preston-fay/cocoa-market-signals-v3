"""
Trade Data Model - UN Comtrade export/import volumes
"""
from sqlmodel import SQLModel, Field
from datetime import datetime, date as date_type
from typing import Optional

class TradeData(SQLModel, table=True):
    """Monthly trade volumes from UN Comtrade"""
    __tablename__ = "trade_data"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Time period (monthly data)
    period: date_type = Field(index=True, description="Month of trade data")
    
    # Trade flow
    flow_type: str = Field(description="Export or Import")
    reporter_country: str = Field(index=True, description="Reporting country")
    partner_country: str = Field(description="Trading partner")
    
    # Volume and value
    quantity_tons: Optional[float] = Field(default=None, description="Trade volume in metric tons")
    trade_value_usd: Optional[float] = Field(default=None, description="Trade value in USD")
    unit_value: Optional[float] = Field(default=None, description="USD per ton")
    
    # Year-over-year changes
    quantity_yoy_change: Optional[float] = Field(default=None, description="YoY change in volume %")
    value_yoy_change: Optional[float] = Field(default=None, description="YoY change in value %")
    
    # Source tracking
    source: str = Field(default="UN Comtrade")
    commodity_code: str = Field(default="1801", description="HS code for cocoa beans")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date_type: lambda v: v.isoformat()
        }