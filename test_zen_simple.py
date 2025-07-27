#!/usr/bin/env python3
"""
Simple test for Zen Consensus with sample real data
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path

from src.models.zen_orchestrator import ZenOrchestrator

def create_sample_real_data():
    """Create sample data from real price statistics"""
    # Based on real data from daily_price_summary_2yr.json
    # min: 3282, max: 12565, mean: 7573, current: 8440
    
    dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
    
    # Create realistic price movement based on real stats
    base_price = 7573  # Real mean
    volatility = 2533  # Real std dev
    
    # Generate price series with realistic volatility
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0, volatility/base_price/np.sqrt(252), len(dates))
    
    prices = [base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        # Keep within realistic bounds
        new_price = max(3282, min(12565, new_price))
        prices.append(new_price)
    
    # Add recent uptrend to match current price
    prices[-30:] = np.linspace(prices[-30], 8440, 30)
    
    df = pd.DataFrame({
        'price': prices,
        'volume': np.random.uniform(1000, 5000, len(dates)),
        'trade_volume_change': np.random.normal(0, 0.1, len(dates))
    }, index=dates)
    
    return df

def test_zen_consensus():
    """Test Zen Consensus with sample data"""
    print("\n" + "="*80)
    print("ZEN CONSENSUS QUICK TEST")
    print("="*80 + "\n")
    
    # Create sample data
    df = create_sample_real_data()
    print(f"Created sample data: {len(df)} days")
    print(f"Price range: ${df['price'].min():.0f} - ${df['price'].max():.0f}")
    print(f"Current price: ${df['price'].iloc[-1]:.0f}")
    
    # Initialize orchestrator
    orchestrator = ZenOrchestrator()
    
    # Run consensus
    print("\n📊 Running Zen Consensus...")
    try:
        result = orchestrator.run_zen_consensus(df)
        consensus = result['consensus']
        
        print(f"\n💵 Current Price: ${consensus['current_price']:,.2f}")
        print(f"🎯 Consensus Forecast: ${consensus['consensus_forecast']:,.2f}")
        print(f"📊 Price Change: {(consensus['consensus_forecast'] - consensus['current_price']) / consensus['current_price'] * 100:+.1f}%")
        print(f"🔮 Confidence Score: {consensus['confidence_score']*100:.0f}%")
        print(f"📈 Signal: {consensus['consensus_signal'].upper()}")
        
        print("\n💡 Recommendations:")
        for rec in consensus['recommendations']:
            print(f"  {rec}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_zen_consensus()