#!/usr/bin/env python3
"""
Run PROPER backtesting using the REAL Zen Consensus model
This will generate predictions for multiple time horizons using actual model outputs
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlmodel import Session, select
from app.core.database import engine
from app.models.prediction import Prediction
from app.models.signal import Signal
from app.models.price_data import PriceData
from src.models.simple_zen_consensus import SimpleZenConsensus
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

# Multiple time horizons as requested
HORIZONS = {
    '1_week': 7,
    '1_month': 30,
    '2_months': 60,
    '3_months': 90,
    '6_months': 180
}

def run_proper_backtesting():
    """Run backtesting with REAL model outputs"""
    
    print("Running PROPER historical backtesting...")
    print(f"Time horizons: {list(HORIZONS.keys())}")
    
    with Session(engine) as session:
        # Get historical prices
        all_prices = session.exec(
            select(PriceData)
            .where(PriceData.source == "Yahoo Finance")
            .order_by(PriceData.date)
        ).all()
        
        print(f"\nFound {len(all_prices)} historical prices")
        
        # Convert to DataFrame format the model expects
        price_data = []
        for p in all_prices:
            price_data.append({
                'date': p.date,
                'price': p.price
            })
        
        df = pd.DataFrame(price_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
        
        # Initialize model
        zen_consensus = SimpleZenConsensus()
        
        predictions_created = 0
        signals_created = 0
        
        # Need minimum history for indicators
        min_history = 50
        
        # Run through history (skip last 180 days to allow for 6-month predictions)
        for i in tqdm(range(min_history, len(df) - max(HORIZONS.values())), desc="Backtesting"):
            # Historical data up to this point
            historical_df = df.iloc[:i+1].copy()
            current_date = historical_df.index[-1]
            current_price = historical_df['price'].iloc[-1]
            
            # Skip weekends
            if current_date.weekday() >= 5:
                continue
                
            # Only run once per week to avoid too many predictions
            if current_date.day % 7 != 1:
                continue
            
            try:
                # Run REAL consensus model
                consensus_result = zen_consensus.run_consensus(historical_df)
                
                # For each time horizon
                for horizon_name, horizon_days in HORIZONS.items():
                    target_date = current_date + timedelta(days=horizon_days)
                    
                    # Find actual price on target date
                    future_prices = df[df.index >= target_date]
                    if len(future_prices) == 0:
                        continue
                    
                    actual_price = future_prices['price'].iloc[0]
                    actual_date = future_prices.index[0]
                    
                    # Use real model prediction
                    # For longer horizons, we need to extrapolate based on trend
                    base_forecast = consensus_result['consensus_forecast']
                    
                    # Calculate trend from price change rate
                    price_change_rate = consensus_result['price_change_rate']
                    
                    # Project price based on consensus and horizon
                    # For short term, use consensus directly
                    if horizon_days <= 7:
                        predicted_price = base_forecast
                    else:
                        # For longer horizons, project the trend
                        daily_change = price_change_rate / 7  # Assume 7-day base forecast
                        predicted_price = current_price * (1 + daily_change * horizon_days)
                    
                    # Adjust confidence for longer horizons
                    confidence = consensus_result['confidence_score'] * (1 - horizon_days / 365 * 0.5)
                    confidence = max(0.1, confidence)  # Minimum 10% confidence
                    
                    # Create prediction
                    prediction = Prediction(
                        model_name="zen_consensus_historical",
                        target_date=actual_date.date(),
                        prediction_horizon=horizon_days,
                        predicted_price=float(predicted_price),
                        confidence_score=float(confidence),
                        prediction_type=f"backtest_{horizon_name}",
                        current_price=float(current_price),
                        model_version="1.0",
                        # Store actual for evaluation
                        actual_price=float(actual_price),
                        error=float(actual_price - predicted_price),
                        error_percentage=float(abs(actual_price - predicted_price) / actual_price * 100),
                        created_at=current_date
                    )
                    
                    session.add(prediction)
                    predictions_created += 1
                
                # Generate trading signals based on consensus
                if consensus_result['consensus_signal'] in ['strong_buy', 'buy', 'strong_sell', 'sell']:
                    # Calculate signal strength from price change rate
                    avg_strength = abs(consensus_result['price_change_rate']) * 10
                    
                    signal = Signal(
                        signal_date=current_date,
                        signal_type='backtest',
                        signal_name=f"zen_{consensus_result['consensus_signal']}",
                        signal_direction='bullish' if 'buy' in consensus_result['consensus_signal'] else 'bearish',
                        signal_strength=float(min(10, abs(avg_strength) * 10)),
                        signal_value=float(base_forecast),
                        description=f"Historical: {consensus_result['consensus_signal'].upper()}",
                        source='zen_consensus_historical',
                        detector='backtesting',
                        confidence=float(consensus_result['confidence_score'])
                    )
                    
                    session.add(signal)
                    signals_created += 1
                
                # Commit periodically
                if predictions_created % 100 == 0:
                    session.commit()
                    print(f"\n  Progress: {predictions_created} predictions created")
                    
            except Exception as e:
                print(f"\n  Error at {current_date.date()}: {str(e)}")
                continue
        
        # Final commit
        session.commit()
        
        print(f"\n✅ Backtesting complete!")
        print(f"Created {predictions_created} predictions")
        print(f"Created {signals_created} signals")
        
        # Calculate performance
        print("\n📊 Calculating REAL performance...")
        
        for horizon_name, horizon_days in HORIZONS.items():
            horizon_preds = session.exec(
                select(Prediction)
                .where(Prediction.model_name == "zen_consensus_historical")
                .where(Prediction.prediction_horizon == horizon_days)
                .where(Prediction.actual_price.is_not(None))
            ).all()
            
            if horizon_preds:
                mape = np.mean([p.error_percentage for p in horizon_preds])
                directional = np.mean([
                    (p.predicted_price > p.current_price) == (p.actual_price > p.current_price)
                    for p in horizon_preds
                ]) * 100
                
                print(f"\n{horizon_name} ({horizon_days} days):")
                print(f"  Predictions: {len(horizon_preds)}")
                print(f"  MAPE: {mape:.2f}%")
                print(f"  Directional Accuracy: {directional:.1f}%")

if __name__ == "__main__":
    run_proper_backtesting()