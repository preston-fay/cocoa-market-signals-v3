#!/usr/bin/env python3
"""
Generate historical predictions to demonstrate actual vs predicted functionality
Uses REAL historical data to create predictions that can be evaluated
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlmodel import Session, select
from app.core.database import engine
from app.models.prediction import Prediction
from app.models.price_data import PriceData
from src.models.simple_zen_consensus import SimpleZenConsensus
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_historical_predictions():
    """Generate predictions for past dates so we can show actual vs predicted"""
    
    print("Generating historical predictions for demonstration...")
    
    with Session(engine) as session:
        # Get historical price data
        prices = session.exec(
            select(PriceData)
            .where(PriceData.source == "Yahoo Finance")
            .order_by(PriceData.date.desc())
            .limit(60)  # Last 60 days
        ).all()
        
        if len(prices) < 30:
            print("Not enough historical data")
            return
        
        # Convert to DataFrame
        price_data = [(p.date, p.price) for p in reversed(prices)]
        df = pd.DataFrame(price_data, columns=['date', 'price'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Initialize consensus model
        consensus = SimpleZenConsensus()
        
        # Generate predictions for dates 30-7 days ago
        predictions_created = 0
        
        for i in range(30, 7, -1):  # From 30 days ago to 7 days ago
            # Use data up to i days ago
            historical_df = df.iloc[:-i]
            
            if len(historical_df) < 20:
                continue
            
            # Run consensus on historical data
            try:
                result = consensus.run_consensus(historical_df)
                
                # Create predictions for 1, 7 days ahead
                current_price = historical_df['price'].iloc[-1]
                prediction_date = historical_df.index[-1].date()
                
                for horizon in [1, 7]:
                    target_date = prediction_date + timedelta(days=horizon)
                    
                    # Only create if we have actual price for that date
                    actual_price_row = df[df.index.date == target_date]
                    if len(actual_price_row) == 0:
                        continue
                    
                    actual_price = actual_price_row['price'].iloc[0]
                    
                    # Adjust prediction based on horizon
                    if horizon == 1:
                        predicted_price = result['consensus_forecast'] * 0.998  # Slight adjustment
                    else:
                        predicted_price = result['consensus_forecast']
                    
                    # Create prediction with actual outcome
                    prediction = Prediction(
                        model_name="zen_consensus",
                        target_date=target_date,
                        prediction_horizon=horizon,
                        predicted_price=predicted_price,
                        confidence_score=result['confidence_score'] * (1 - horizon * 0.01),
                        prediction_type="historical_demo",
                        current_price=current_price,
                        model_version="1.0",
                        # Fill in actual outcome
                        actual_price=actual_price,
                        error=actual_price - predicted_price,
                        error_percentage=abs(actual_price - predicted_price) / actual_price * 100
                    )
                    
                    # Override created_at to be historical
                    prediction.created_at = datetime.combine(prediction_date, datetime.min.time())
                    
                    session.add(prediction)
                    predictions_created += 1
                    
                    print(f"  Created prediction: {prediction_date} -> {target_date} "
                          f"(predicted: ${predicted_price:,.0f}, actual: ${actual_price:,.0f})")
                
            except Exception as e:
                print(f"  Error for date {i} days ago: {e}")
                continue
        
        session.commit()
        print(f"\nCreated {predictions_created} historical predictions with outcomes")
        
        # Show summary
        all_predictions = session.exec(
            select(Prediction)
            .where(Prediction.model_name == "zen_consensus")
            .where(Prediction.actual_price.is_not(None))
        ).all()
        
        if all_predictions:
            mape = np.mean([p.error_percentage for p in all_predictions])
            directional = np.mean([
                (p.predicted_price > p.current_price) == (p.actual_price > p.current_price)
                for p in all_predictions
            ])
            
            print(f"\nOverall Statistics:")
            print(f"  Total predictions with outcomes: {len(all_predictions)}")
            print(f"  Mean Absolute Percentage Error: {mape:.2f}%")
            print(f"  Directional Accuracy: {directional:.1%}")

if __name__ == "__main__":
    generate_historical_predictions()