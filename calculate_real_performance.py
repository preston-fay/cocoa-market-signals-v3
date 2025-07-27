#!/usr/bin/env python3
"""
Calculate REAL performance metrics from actual predictions in database
NO FAKE DATA - only real predictions vs actual outcomes
"""
from sqlmodel import Session, select, and_
from app.core.database import engine
from app.models.prediction import Prediction
from app.models.price_data import PriceData
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def calculate_real_performance():
    """Calculate real performance metrics from database"""
    
    with Session(engine) as session:
        # Get all predictions with actual outcomes
        predictions = session.exec(
            select(Prediction)
            .where(Prediction.actual_price.is_not(None))
            .order_by(Prediction.created_at)
        ).all()
        
        print(f"Found {len(predictions)} predictions with actual outcomes")
        
        if not predictions:
            print("No predictions with outcomes found!")
            return
        
        # Group by prediction horizon
        horizons = {}
        for pred in predictions:
            horizon = pred.prediction_horizon
            if horizon not in horizons:
                horizons[horizon] = []
            horizons[horizon].append(pred)
        
        print("\n📊 REAL Performance Metrics by Horizon:")
        print("-" * 60)
        
        overall_errors = []
        overall_directional = []
        
        for horizon, preds in sorted(horizons.items()):
            print(f"\n{horizon}-day predictions:")
            print(f"  Count: {len(preds)}")
            
            # Calculate REAL metrics
            errors = []
            directional_correct = []
            
            for pred in preds:
                # Absolute percentage error
                error_pct = abs(pred.actual_price - pred.predicted_price) / pred.actual_price * 100
                errors.append(error_pct)
                
                # Directional accuracy
                pred_direction = pred.predicted_price > pred.current_price
                actual_direction = pred.actual_price > pred.current_price
                directional_correct.append(pred_direction == actual_direction)
                
                overall_errors.append(error_pct)
                overall_directional.append(pred_direction == actual_direction)
            
            # Real statistics
            mape = np.mean(errors)
            rmse = np.sqrt(np.mean([(p.actual_price - p.predicted_price)**2 for p in preds]))
            directional_acc = np.mean(directional_correct) * 100
            
            print(f"  MAPE: {mape:.2f}%")
            print(f"  RMSE: ${rmse:,.2f}")
            print(f"  Directional Accuracy: {directional_acc:.1f}%")
            print(f"  Best prediction error: {min(errors):.2f}%")
            print(f"  Worst prediction error: {max(errors):.2f}%")
        
        # Overall metrics
        print("\n📈 Overall Performance:")
        print("-" * 60)
        print(f"Total predictions evaluated: {len(predictions)}")
        print(f"Mean Absolute Percentage Error: {np.mean(overall_errors):.2f}%")
        print(f"Overall Directional Accuracy: {np.mean(overall_directional) * 100:.1f}%")
        
        # Performance over time
        print("\n📅 Performance by Month:")
        print("-" * 60)
        
        # Group by month
        monthly_data = {}
        for pred in predictions:
            month_key = pred.created_at.strftime("%Y-%m")
            if month_key not in monthly_data:
                monthly_data[month_key] = []
            
            error_pct = abs(pred.actual_price - pred.predicted_price) / pred.actual_price * 100
            monthly_data[month_key].append(error_pct)
        
        for month, errors in sorted(monthly_data.items()):
            print(f"{month}: MAPE = {np.mean(errors):.2f}% (n={len(errors)})")
        
        # Check for any signals
        from app.models.signal import Signal
        signals = session.exec(
            select(Signal)
            .where(Signal.source == "zen_consensus")
            .limit(10)
        ).all()
        
        print(f"\n📡 Signals found: {len(signals)}")
        
        return predictions

if __name__ == "__main__":
    calculate_real_performance()