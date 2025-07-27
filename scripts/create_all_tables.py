#!/usr/bin/env python3
"""
Create all database tables
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlmodel import SQLModel
from app.core.database import engine

# Import all models to register them
from app.models.price_data import PriceData
from app.models.prediction import Prediction
from app.models.signal import Signal
from app.models.model_performance import ModelPerformance
from app.models.trade_data import TradeData
from app.models.weather_data import WeatherData
from app.models.news_article import NewsArticle

def create_tables():
    """Create all tables in the database"""
    print("Creating all database tables...")
    
    # Create tables
    SQLModel.metadata.create_all(engine)
    
    print("✓ All tables created successfully")
    
    # List tables
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    print(f"\nDatabase contains {len(tables)} tables:")
    for table in sorted(tables):
        print(f"  - {table}")

if __name__ == "__main__":
    create_tables()