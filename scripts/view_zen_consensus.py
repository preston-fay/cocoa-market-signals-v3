#!/usr/bin/env python3
"""
View Current Zen Consensus Predictions and Signals
Shows the latest consensus from the database
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlmodel import Session, select
from app.core.database import engine
from app.models.prediction import Prediction
from app.models.signal import Signal
from datetime import datetime, date, timedelta
import pandas as pd

def view_latest_consensus():
    """Display latest Zen Consensus predictions and signals"""
    print("\n" + "="*60)
    print("CURRENT ZEN CONSENSUS")
    print("="*60)
    
    with Session(engine) as session:
        # Get latest predictions
        latest_predictions = session.exec(
            select(Prediction)
            .where(Prediction.model_name == "zen_consensus")
            .order_by(Prediction.created_at.desc())
            .limit(3)
        ).all()
        
        if latest_predictions:
            print("\n📊 LATEST PREDICTIONS:")
            print("-" * 40)
            
            for pred in latest_predictions:
                days_ahead = (pred.target_date - date.today()).days
                price_change = (pred.predicted_price - pred.current_price) / pred.current_price * 100
                
                print(f"\n{days_ahead}-Day Forecast:")
                print(f"  Target Date: {pred.target_date}")
                print(f"  Current Price: ${pred.current_price:,.0f}")
                print(f"  Predicted Price: ${pred.predicted_price:,.0f}")
                print(f"  Expected Change: {price_change:+.1f}%")
                print(f"  Confidence: {pred.confidence_score:.1%}")
        
        # Get latest signals
        latest_signals = session.exec(
            select(Signal)
            .where(Signal.source == "zen_consensus")
            .order_by(Signal.detected_at.desc())
            .limit(1)
        ).all()
        
        if latest_signals:
            print("\n🎯 CURRENT SIGNAL:")
            print("-" * 40)
            
            signal = latest_signals[0]
            print(f"\nSignal: {signal.signal_name.upper()}")
            print(f"Direction: {signal.signal_direction}")
            print(f"Strength: {signal.signal_strength:.1f} (scale: -10 to +10)")
            print(f"Confidence: {signal.confidence:.1%}")
            print(f"Description: {signal.description}")
            print(f"Generated: {signal.detected_at.strftime('%Y-%m-%d %H:%M UTC')}")
        
        # Get role signals
        role_signals = session.exec(
            select(Signal)
            .where(Signal.detector == "simple_zen_consensus")
            .where(Signal.signal_type == "role")
            .order_by(Signal.detected_at.desc())
            .limit(3)
        ).all()
        
        if role_signals:
            print("\n🤖 ROLE PERSPECTIVES:")
            print("-" * 40)
            
            for signal in role_signals:
                print(f"\n{signal.source}:")
                print(f"  Signal: {signal.signal_direction}")
                print(f"  Strength: {signal.signal_strength:+.1f}")
                print(f"  Confidence: {signal.confidence:.1%}")
        
        # Get market warnings
        warnings = session.exec(
            select(Signal)
            .where(Signal.signal_type == "volatility")
            .order_by(Signal.detected_at.desc())
            .limit(1)
        ).all()
        
        if warnings:
            print("\n⚠️  MARKET WARNINGS:")
            print("-" * 40)
            
            for warning in warnings:
                print(f"\n{warning.description}")
                print(f"Volatility Level: {warning.signal_value:.0f}%")
    
    print("\n" + "="*60)
    print("💡 Interpretation: The Zen Consensus combines multiple")
    print("   AI perspectives to reach a balanced market view.")
    print("="*60)

if __name__ == "__main__":
    view_latest_consensus()