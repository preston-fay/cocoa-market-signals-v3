#!/usr/bin/env python3
"""
Run REAL historical backtesting using actual models
NO FAKE DATA - Uses real model predictions on historical data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlmodel import Session, select, and_
from app.core.database import engine
from app.models.prediction import Prediction
from app.models.signal import Signal
from app.models.price_data import PriceData
from app.models.model_performance import ModelPerformance
from src.models.simple_zen_consensus import SimpleZenConsensus
# Statistical models not needed for basic backtesting
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Prediction horizons in days
HORIZONS = {
    '1_week': 7,
    '1_month': 30,
    '2_months': 60,
    '3_months': 90,
    '6_months': 180
}

def run_real_historical_backtesting():
    """
    Run REAL backtesting using actual models on historical data
    This simulates what would have happened if we ran our models
    throughout history
    """
    
    print("Running REAL historical backtesting with actual models...")
    print(f"Prediction horizons: {list(HORIZONS.keys())}")
    
    with Session(engine) as session:
        # Get all historical price data
        all_prices = session.exec(
            select(PriceData)
            .where(PriceData.source == "Yahoo Finance")
            .order_by(PriceData.date)
        ).all()
        
        if len(all_prices) < 100:
            print("Not enough historical data for backtesting")
            return
        
        # Convert to DataFrame
        price_records = [(p.date, p.price) for p in all_prices]
        df_prices = pd.DataFrame(price_records, columns=['date', 'price'])
        df_prices['date'] = pd.to_datetime(df_prices['date'])
        df_prices.set_index('date', inplace=True)
        
        print(f"\nHistorical data range: {df_prices.index[0].date()} to {df_prices.index[-1].date()}")
        print(f"Total days: {len(df_prices)}")
        
        # Initialize real models
        zen_consensus = SimpleZenConsensus()
        
        predictions_created = 0
        signals_created = 0
        
        # Need enough history for models to work properly
        min_history = 60
        
        # Run backtesting through history
        print("\nRunning models through history...")
        for i in tqdm(range(min_history, len(df_prices) - max(HORIZONS.values())), desc="Backtesting"):
            # Get historical data up to this point
            historical_data = df_prices.iloc[:i+1].copy()
            current_date = historical_data.index[-1].date()
            current_price = historical_data['price'].iloc[-1]
            
            # Skip if we already have predictions for this date
            existing = session.exec(
                select(Prediction)
                .where(and_(
                    Prediction.model_name == "zen_consensus_backtest",
                    Prediction.created_at >= datetime.combine(current_date, datetime.min.time()),
                    Prediction.created_at < datetime.combine(current_date + timedelta(days=1), datetime.min.time())
                ))
                .limit(1)
            ).first()
            
            if existing:
                continue
            
            try:
                # Run Zen Consensus with real indicators
                consensus_result = zen_consensus.run_consensus(historical_data)
                
                # For each prediction horizon
                for horizon_name, horizon_days in HORIZONS.items():
                    target_date = current_date + timedelta(days=horizon_days)
                    
                    # Check if we have actual future price
                    future_idx = i + horizon_days
                    if future_idx >= len(df_prices):
                        continue
                        
                    actual_future_price = df_prices.iloc[future_idx]['price']
                    
                    # Use real model predictions
                    # Zen Consensus provides the base prediction
                    base_prediction = consensus_result['consensus_forecast']
                    
                    # For longer horizons, use the trend from consensus
                    # Real models don't predict far into future, so we use trend extrapolation
                    trend = consensus_result['predictions']['trend_follower']['prediction']
                    if trend > current_price:
                        # Bullish trend
                        daily_change = (trend - current_price) / 7  # 7-day trend
                        predicted_price = current_price + (daily_change * horizon_days)
                    else:
                        # Use base prediction with decay for uncertainty
                        predicted_price = base_prediction
                    
                    # Calculate real confidence based on model agreement
                    confidence = consensus_result['confidence_score']
                    if horizon_days > 30:
                        confidence *= 0.8  # Lower confidence for longer horizons
                    
                    # Create prediction record with REAL model output
                    prediction = Prediction(
                        model_name="zen_consensus_backtest",
                        target_date=target_date,
                        prediction_horizon=horizon_days,
                        predicted_price=float(predicted_price),
                        confidence_score=float(confidence),
                        prediction_type=f"historical_{horizon_name}",
                        current_price=float(current_price),
                        model_version="1.0",
                        # Store actual outcome for evaluation
                        actual_price=float(actual_future_price),
                        error=float(actual_future_price - predicted_price),
                        error_percentage=float(abs(actual_future_price - predicted_price) / actual_future_price * 100),
                        created_at=datetime.combine(current_date, datetime.min.time())
                    )
                    
                    session.add(prediction)
                    predictions_created += 1
                
                # Create REAL trading signals based on model consensus
                if consensus_result['consensus_signal'] in ['strong_buy', 'buy', 'strong_sell', 'sell']:
                    # Real signal strength from models
                    signal_strength = consensus_result['weights']['trend_follower'] * \
                                    consensus_result['predictions']['trend_follower']['signal_strength']
                    
                    signal = Signal(
                        signal_date=datetime.combine(current_date, datetime.min.time()),
                        signal_type='backtest',
                        signal_name=f"zen_{consensus_result['consensus_signal']}",
                        signal_direction='bullish' if 'buy' in consensus_result['consensus_signal'] else 'bearish',
                        signal_strength=float(min(10, abs(signal_strength) * 10)),
                        signal_value=float(predicted_price),
                        description=f"Backtest: {consensus_result['consensus_signal'].upper()} signal",
                        source='zen_consensus_backtest',
                        detector='historical_backtesting',
                        confidence=float(consensus_result['confidence_score'])
                    )
                    
                    session.add(signal)
                    signals_created += 1
                
                # Commit periodically
                if predictions_created % 50 == 0:
                    session.commit()
                    
            except Exception as e:
                import traceback
                if predictions_created == 0:  # Only print full trace for first error
                    print(f"\nDetailed error at {current_date}:")
                    traceback.print_exc()
                else:
                    print(f"\nError at {current_date}: {str(e)}")
                continue
        
        # Final commit
        session.commit()
        
        print(f"\n✅ Real backtesting complete!")
        print(f"Created {predictions_created} predictions using REAL models")
        print(f"Created {signals_created} trading signals")
        
        # Calculate REAL performance metrics
        print("\n📊 REAL Model Performance by Horizon:")
        
        for horizon_name, horizon_days in HORIZONS.items():
            horizon_preds = session.exec(
                select(Prediction)
                .where(and_(
                    Prediction.model_name == "zen_consensus_backtest",
                    Prediction.prediction_horizon == horizon_days,
                    Prediction.actual_price.is_not(None)
                ))
            ).all()
            
            if horizon_preds:
                mape = np.mean([p.error_percentage for p in horizon_preds])
                rmse = np.sqrt(np.mean([p.error**2 for p in horizon_preds]))
                directional = np.mean([
                    (p.predicted_price > p.current_price) == (p.actual_price > p.current_price)
                    for p in horizon_preds
                ])
                
                print(f"\n{horizon_name} ({horizon_days} days):")
                print(f"  Total predictions: {len(horizon_preds)}")
                print(f"  MAPE: {mape:.2f}%")
                print(f"  RMSE: ${rmse:,.2f}")
                print(f"  Directional Accuracy: {directional:.1%}")
                
                # Store performance metrics
                perf = ModelPerformance(
                    model_name="zen_consensus_backtest",
                    metric_name=f"mape_{horizon_name}",
                    metric_value=float(mape),
                    evaluation_date=date.today(),
                    prediction_horizon=horizon_days,
                    sample_size=len(horizon_preds)
                )
                session.add(perf)
        
        session.commit()
        
        # Analyze trading signals
        all_signals = session.exec(
            select(Signal)
            .where(Signal.source == "zen_consensus_backtest")
        ).all()
        
        print(f"\n📈 Trading Signal Analysis:")
        print(f"  Total signals: {len(all_signals)}")
        
        buy_signals = [s for s in all_signals if 'buy' in s.signal_name]
        sell_signals = [s for s in all_signals if 'sell' in s.signal_name]
        
        print(f"  Buy signals: {len(buy_signals)}")
        print(f"  Sell signals: {len(sell_signals)}")
        
        # Check signal accuracy
        print("\n🎯 Signal Accuracy (checking 1-week ahead):")
        correct_signals = 0
        total_checked = 0
        
        for signal in all_signals[:100]:  # Check first 100 signals
            signal_date = signal.signal_date.date() if hasattr(signal.signal_date, 'date') else signal.signal_date
            
            # Get price 1 week later
            future_price = session.exec(
                select(PriceData)
                .where(and_(
                    PriceData.source == "Yahoo Finance",
                    PriceData.date >= signal_date + timedelta(days=7),
                    PriceData.date <= signal_date + timedelta(days=14)
                ))
                .order_by(PriceData.date)
                .limit(1)
            ).first()
            
            if future_price:
                current_price = signal.signal_value
                if 'buy' in signal.signal_name and future_price.price > current_price:
                    correct_signals += 1
                elif 'sell' in signal.signal_name and future_price.price < current_price:
                    correct_signals += 1
                total_checked += 1
        
        if total_checked > 0:
            print(f"  Signal accuracy (1-week): {correct_signals/total_checked:.1%}")

if __name__ == "__main__":
    run_real_historical_backtesting()