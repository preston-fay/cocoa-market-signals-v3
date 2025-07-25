"""
Test Model Orchestration with Dynamic Model Selection
Shows how models are selected based on market regime
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import orchestrator
from src.models.model_orchestrator import ModelOrchestrator

print("="*80)
print("TESTING MODEL ORCHESTRATION WITH DYNAMIC SELECTION")
print("="*80)

# Load real unified data
print("\n1. LOADING REAL DATA")
print("-" * 60)
df = pd.read_csv("data/processed/unified_real_data.csv")
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
df.set_index('date', inplace=True)
print(f"✓ Loaded {len(df)} days of unified REAL data")

# Initialize orchestrator
orchestrator = ModelOrchestrator()

# Run orchestrated analysis
print("\n2. RUNNING ORCHESTRATED ANALYSIS")
print("-" * 60)

results = orchestrator.run_orchestrated_analysis(df)

# Display results
print("\n" + "="*80)
print("ORCHESTRATION RESULTS")
print("="*80)

print("\n1. MARKET REGIME DETECTION:")
print("-" * 40)
regime_info = results['regime']
print(f"Current Regime: {regime_info['regime'].upper()}")
print(f"Volatility: {regime_info['volatility']:.2f}% (percentile: {regime_info['volatility_percentile']:.0f}%)")
print(f"5-day Momentum: {regime_info['momentum_5d']:+.2f}%")
print(f"20-day Momentum: {regime_info['momentum_20d']:+.2f}%")

print("\n2. MODELS SELECTED:")
print("-" * 40)
print(f"Forecast Models: {', '.join(results['models_used']['forecast'])}")
print(f"Anomaly Models: {', '.join(results['models_used']['anomaly'])}")

print("\n3. TRADING SIGNALS:")
print("-" * 40)
signals = results['signals']
print(f"Direction: {signals['direction'].upper()}")
print(f"Signal Strength: {signals['signal_strength']:.2%}")
print(f"Confidence: {signals['confidence']:.2%}")
print(f"7-day Price Forecast: ${signals['price_forecast']:,.0f}" if signals['price_forecast'] else "No forecast available")
print(f"Current Price: ${df['price'].iloc[-1]:,.0f}")
if signals['price_forecast']:
    change_pct = (signals['price_forecast'] - df['price'].iloc[-1]) / df['price'].iloc[-1] * 100
    print(f"Expected Change: {change_pct:+.2f}%")
print(f"Risk Level: {signals['risk_level'].upper()}")

print("\n4. MODEL PERFORMANCE:")
print("-" * 40)
for model_name, model_result in results['forecasts'].items():
    if 'mape' in model_result:
        print(f"{model_name.upper()}: MAPE = {model_result['mape']:.2f}%")
    elif 'cv_scores' in model_result:
        print(f"{model_name.upper()}: CV MAPE = {np.mean(model_result['cv_scores']):.2f}%")

print("\n5. ANOMALY DETECTION:")
print("-" * 40)
for model_name, anomaly_result in results['anomalies'].items():
    if 'anomaly_dates' in anomaly_result:
        count = len(anomaly_result['anomaly_dates'])
        print(f"{model_name.upper()}: {count} anomalies detected")
        if count > 0 and hasattr(anomaly_result['anomaly_dates'], '__iter__'):
            # Show recent anomalies
            recent = [date for date in anomaly_result['anomaly_dates'] 
                     if (pd.Timestamp.now() - pd.Timestamp(date)).days < 30]
            if recent:
                print(f"  Recent (last 30 days): {len(recent)} anomalies")

print("\n6. RECOMMENDATIONS:")
print("-" * 40)
for i, rec in enumerate(results['recommendations'], 1):
    print(f"{i}. {rec}")

# Run backtest
print("\n\n" + "="*80)
print("BACKTESTING ORCHESTRATION STRATEGY")
print("="*80)

print("\nRunning 90-day backtest...")
backtest_results = orchestrator.backtest_orchestration(df, test_period_days=90)

print("\nBacktest Results:")
print("-" * 40)
print(f"Prediction MAPE: {backtest_results['mape']:.2f}%")
print(f"Cumulative Returns: {backtest_results['cumulative_returns']:+.2f}%")
print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
print(f"\nRegime Distribution:")
for regime, count in backtest_results['regime_distribution'].items():
    print(f"  {regime}: {count} days ({count/sum(backtest_results['regime_distribution'].values())*100:.1f}%)")

# Save orchestration results
orchestration_summary = {
    "timestamp": datetime.now().isoformat(),
    "current_regime": regime_info['regime'],
    "volatility": float(regime_info['volatility']),
    "models_selected": results['models_used'],
    "signal": {
        "direction": signals['direction'],
        "strength": float(signals['signal_strength']),
        "confidence": float(signals['confidence']),
        "forecast": float(signals['price_forecast']) if signals['price_forecast'] else None,
        "risk_level": signals['risk_level']
    },
    "backtest": {
        "mape": float(backtest_results['mape']),
        "cumulative_returns": float(backtest_results['cumulative_returns']),
        "sharpe_ratio": float(backtest_results['sharpe_ratio'])
    },
    "recommendations": results['recommendations']
}

with open("data/processed/orchestration_results.json", "w") as f:
    json.dump(orchestration_summary, f, indent=2)

print("\n✓ Orchestration results saved to data/processed/orchestration_results.json")

# Create actual vs predicted data for visualization
if backtest_results['results_df'] is not None and len(backtest_results['results_df']) > 0:
    viz_data = backtest_results['results_df'][['dates', 'actual_prices', 'predicted_prices', 'regimes']].copy()
    viz_data.to_csv("data/processed/orchestration_backtest_viz.csv", index=False)
    print("✓ Visualization data saved to data/processed/orchestration_backtest_viz.csv")

print("\n" + "="*80)
print("ORCHESTRATION COMPLETE - READY FOR PRODUCTION!")
print("="*80)