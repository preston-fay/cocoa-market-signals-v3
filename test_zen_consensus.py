#!/usr/bin/env python3
"""
Test Zen Consensus Model Orchestrator with REAL data
NO FAKE DATA - 100% real market analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

from src.models.zen_orchestrator import ZenOrchestrator
from src.data_pipeline.unified_pipeline_real import UnifiedDataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_real_data():
    """Load real cocoa market data"""
    logger.info("Loading REAL cocoa market data...")
    
    # Try to load from processed data first
    processed_file = Path("data/processed/cocoa_combined_data.csv")
    if processed_file.exists():
        df = pd.read_csv(processed_file, index_col='date', parse_dates=True)
        logger.info(f"Loaded {len(df)} days of real data from {df.index[0]} to {df.index[-1]}")
        return df
    
    # Otherwise fetch fresh data
    logger.info("Fetching fresh data from sources...")
    pipeline = UnifiedDataPipeline()
    df = pipeline.prepare_unified_dataset()
    
    if df is not None and not df.empty:
        logger.info(f"Fetched {len(df)} days of fresh data")
        return df
    else:
        raise ValueError("Could not load real data!")

def run_zen_consensus_test():
    """Test Zen Consensus with real data"""
    print("\n" + "="*80)
    print("ZEN CONSENSUS MODEL ORCHESTRATOR TEST")
    print("Using 100% REAL cocoa market data - NO synthetic data")
    print("="*80 + "\n")
    
    # Load data
    df = load_real_data()
    
    # Initialize orchestrator
    orchestrator = ZenOrchestrator()
    
    # Get latest consensus
    print("\n📊 Running Zen Consensus Analysis...")
    print("-" * 60)
    
    consensus_result = orchestrator.run_zen_consensus(df)
    consensus = consensus_result['consensus']
    role_results = consensus_result['role_results']
    
    # Display results
    print(f"\n💵 Current Price: ${consensus['current_price']:,.2f}")
    print(f"🎯 Consensus Forecast: ${consensus['consensus_forecast']:,.2f}")
    print(f"📊 Price Change: {(consensus['consensus_forecast'] - consensus['current_price']) / consensus['current_price'] * 100:+.1f}%")
    print(f"🔮 Confidence Score: {consensus['confidence_score']*100:.0f}%")
    print(f"📈 Signal: {consensus['consensus_signal'].upper()}")
    
    # Role contributions
    print("\n🧠 Model Role Contributions:")
    print("-" * 60)
    for role, contrib in consensus['role_contributions'].items():
        print(f"{role:25} | Prediction: ${contrib['prediction']:,.0f} ({contrib['change_pct']:+.1f}%) | Confidence: {contrib['confidence']*100:.0f}%")
    
    # Market context
    print("\n📊 Market Context:")
    print("-" * 60)
    context = consensus['market_context']
    print(f"5-Day Trend:  {context['trend_5d']:+.1f}%")
    print(f"20-Day Trend: {context['trend_20d']:+.1f}%")
    print(f"Volatility:   {context['volatility_20d']:.1f}%")
    print(f"Price Level:  {context['price_percentile']:.0f}th percentile")
    
    # Recommendations
    print("\n💡 Recommendations:")
    print("-" * 60)
    for rec in consensus['recommendations']:
        print(f"  {rec}")
    
    # Dissenting views
    if consensus['dissenting_views']:
        print("\n⚠️ Dissenting Views:")
        print("-" * 60)
        for dissent in consensus['dissenting_views']:
            print(f"  {dissent}")
    
    # Run backtest
    print("\n\n📈 Running Backtest (90 days)...")
    print("-" * 60)
    
    backtest_results = orchestrator.backtest_zen_consensus(df, test_days=90)
    
    print(f"\nBacktest Performance:")
    print(f"  • MAPE: {backtest_results['mape']:.2f}%")
    print(f"  • Cumulative Return: {backtest_results['cumulative_return_pct']:+.1f}%")
    print(f"  • Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"  • Avg Confidence: {backtest_results['avg_confidence']*100:.0f}%")
    
    print(f"\nSignal Distribution:")
    for signal, count in backtest_results['signal_distribution'].items():
        print(f"  • {signal}: {count} days")
    
    # Save results
    results = {
        'test_date': datetime.now().isoformat(),
        'consensus': {
            'current_price': float(consensus['current_price']),
            'forecast': float(consensus['consensus_forecast']),
            'signal': consensus['consensus_signal'],
            'confidence': float(consensus['confidence_score']),
            'role_contributions': {
                role: {
                    'prediction': float(contrib['prediction']),
                    'change_pct': float(contrib['change_pct']),
                    'confidence': float(contrib['confidence'])
                }
                for role, contrib in consensus['role_contributions'].items()
            }
        },
        'backtest': {
            'mape': float(backtest_results['mape']),
            'cumulative_return_pct': float(backtest_results['cumulative_return_pct']),
            'sharpe_ratio': float(backtest_results['sharpe_ratio']),
            'avg_confidence': float(backtest_results['avg_confidence']),
            'signal_distribution': backtest_results['signal_distribution']
        },
        'market_context': {k: float(v) for k, v in context.items() if isinstance(v, (int, float))}
    }
    
    output_file = Path("data/processed/zen_consensus_results.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Save backtest details
    backtest_df = backtest_results['results_df']
    backtest_file = Path("data/processed/zen_consensus_backtest.csv")
    backtest_df.to_csv(backtest_file)
    print(f"📊 Backtest details saved to: {backtest_file}")
    
    print("\n" + "="*80)
    print("ZEN CONSENSUS TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_zen_consensus_test()