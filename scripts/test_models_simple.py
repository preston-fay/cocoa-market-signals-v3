#!/usr/bin/env python3
"""
Simplified Model Testing - Clear Results with Real Data
Focuses on models that work well with price data only
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from sqlmodel import Session, select
from app.core.database import engine
from app.models.price_data import PriceData
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_price_data():
    """Load real price data from database"""
    print("\nLoading real cocoa price data...")
    
    with Session(engine) as session:
        prices = session.exec(
            select(PriceData)
            .where(PriceData.source == "Yahoo Finance")
            .order_by(PriceData.date)
        ).all()
        
        df = pd.DataFrame([
            {'date': p.date, 'price': p.price}
            for p in prices
        ])
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        print(f"✓ Loaded {len(df)} days of data")
        print(f"  Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
        print(f"  Current price: ${df['price'].iloc[-1]:,.0f}")
        
        return df

def calculate_mape(actual, predicted):
    """Calculate MAPE handling edge cases"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Avoid division by zero
    mask = actual != 0
    if not mask.any():
        return np.nan
        
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def test_simple_models(df, test_days=30):
    """Test simple but effective models"""
    # Split data
    train = df[:-test_days]
    test = df[-test_days:]
    
    results = {}
    
    print("\n" + "="*60)
    print("TESTING SIMPLE PREDICTIVE MODELS")
    print("="*60)
    print(f"\nTraining on {len(train)} days, testing on {len(test)} days")
    
    # 1. Simple Moving Average
    print("\n1. SIMPLE MOVING AVERAGE (SMA)")
    print("-" * 30)
    
    periods = [5, 10, 20, 30]
    sma_results = {}
    
    for period in periods:
        sma = train['price'].rolling(period).mean().iloc[-1]
        predictions = np.full(len(test), sma)
        
        mae = mean_absolute_error(test['price'], predictions)
        mape = calculate_mape(test['price'], predictions)
        
        sma_results[f'SMA_{period}'] = {
            'prediction': sma,
            'mae': mae,
            'mape': mape
        }
        
        print(f"  SMA-{period}: ${sma:,.0f} (MAPE: {mape:.1f}%)")
    
    # Find best SMA
    best_sma = min(sma_results.items(), key=lambda x: x[1]['mape'] if not np.isnan(x[1]['mape']) else float('inf'))
    results['best_sma'] = {'model': best_sma[0], **best_sma[1]}
    print(f"\n  Best: {best_sma[0]} with {best_sma[1]['mape']:.1f}% error")
    
    # 2. Exponential Moving Average
    print("\n2. EXPONENTIAL MOVING AVERAGE (EMA)")
    print("-" * 30)
    
    spans = [10, 20, 30, 50]
    ema_results = {}
    
    for span in spans:
        ema = train['price'].ewm(span=span, adjust=False).mean().iloc[-1]
        predictions = np.full(len(test), ema)
        
        mae = mean_absolute_error(test['price'], predictions)
        mape = calculate_mape(test['price'], predictions)
        
        ema_results[f'EMA_{span}'] = {
            'prediction': ema,
            'mae': mae,
            'mape': mape
        }
        
        print(f"  EMA-{span}: ${ema:,.0f} (MAPE: {mape:.1f}%)")
    
    best_ema = min(ema_results.items(), key=lambda x: x[1]['mape'] if not np.isnan(x[1]['mape']) else float('inf'))
    results['best_ema'] = {'model': best_ema[0], **best_ema[1]}
    print(f"\n  Best: {best_ema[0]} with {best_ema[1]['mape']:.1f}% error")
    
    # 3. Linear Trend
    print("\n3. LINEAR TREND EXTRAPOLATION")
    print("-" * 30)
    
    # Fit linear trend on last N days
    trend_periods = [30, 60, 90, 120]
    trend_results = {}
    
    for period in trend_periods:
        if period > len(train):
            continue
            
        recent_data = train[-period:]
        x = np.arange(len(recent_data))
        y = recent_data['price'].values
        
        # Fit linear regression
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        
        # Predict for test period
        test_x = np.arange(len(recent_data), len(recent_data) + len(test))
        predictions = slope * test_x + intercept
        
        mae = mean_absolute_error(test['price'], predictions)
        mape = calculate_mape(test['price'], predictions)
        
        trend_results[f'Trend_{period}d'] = {
            'slope': slope,
            'last_prediction': predictions[-1],
            'mae': mae,
            'mape': mape
        }
        
        print(f"  {period}-day trend: Slope=${slope:.2f}/day (MAPE: {mape:.1f}%)")
    
    if trend_results:
        best_trend = min(trend_results.items(), key=lambda x: x[1]['mape'] if not np.isnan(x[1]['mape']) else float('inf'))
        results['best_trend'] = {'model': best_trend[0], **best_trend[1]}
        print(f"\n  Best: {best_trend[0]} with {best_trend[1]['mape']:.1f}% error")
    
    # 4. Naive Models (Benchmarks)
    print("\n4. NAIVE BENCHMARKS")
    print("-" * 30)
    
    # Last value (persistence)
    last_value = train['price'].iloc[-1]
    predictions = np.full(len(test), last_value)
    mae = mean_absolute_error(test['price'], predictions)
    mape = calculate_mape(test['price'], predictions)
    
    results['naive_last'] = {
        'prediction': last_value,
        'mae': mae,
        'mape': mape
    }
    print(f"  Last Value: ${last_value:,.0f} (MAPE: {mape:.1f}%)")
    
    # Average of all historical
    avg_all = train['price'].mean()
    predictions = np.full(len(test), avg_all)
    mae = mean_absolute_error(test['price'], predictions)
    mape = calculate_mape(test['price'], predictions)
    
    results['naive_mean'] = {
        'prediction': avg_all,
        'mae': mae,
        'mape': mape
    }
    print(f"  Historical Mean: ${avg_all:,.0f} (MAPE: {mape:.1f}%)")
    
    # 5. Volatility-Adjusted Prediction
    print("\n5. VOLATILITY-ADJUSTED FORECAST")
    print("-" * 30)
    
    # Calculate recent volatility
    returns = train['price'].pct_change().dropna()
    recent_vol = returns[-30:].std()
    annual_vol = recent_vol * np.sqrt(252)
    
    # Base prediction with volatility bands
    base = train['price'].iloc[-1]
    days_ahead = len(test)
    vol_adjustment = base * recent_vol * np.sqrt(days_ahead)
    
    upper_band = base + vol_adjustment
    lower_band = base - vol_adjustment
    
    print(f"  Base: ${base:,.0f}")
    print(f"  30-day volatility: {annual_vol*100:.1f}% annualized")
    print(f"  {days_ahead}-day range: ${lower_band:,.0f} - ${upper_band:,.0f}")
    
    # Check how often price stayed within bands
    within_bands = ((test['price'] >= lower_band) & (test['price'] <= upper_band)).sum()
    band_accuracy = within_bands / len(test) * 100
    
    results['volatility_bands'] = {
        'base': base,
        'annual_volatility': annual_vol,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'band_accuracy': band_accuracy
    }
    print(f"  Band accuracy: {band_accuracy:.1f}% of days within bands")
    
    return results, train, test

def analyze_errors(results, test_df):
    """Analyze prediction errors"""
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    # Rank models by MAPE
    model_ranking = []
    
    for category in ['best_sma', 'best_ema', 'best_trend', 'naive_last', 'naive_mean']:
        if category in results and 'mape' in results[category]:
            model_ranking.append({
                'model': results[category].get('model', category),
                'mape': results[category]['mape'],
                'mae': results[category]['mae']
            })
    
    model_ranking.sort(key=lambda x: x['mape'] if not np.isnan(x['mape']) else float('inf'))
    
    print("\nModel Rankings (by MAPE):")
    print("-" * 40)
    for i, model in enumerate(model_ranking[:5]):
        print(f"{i+1}. {model['model']}: {model['mape']:.1f}% error (${model['mae']:.0f} MAE)")
    
    # Price movement analysis
    actual_change = test_df['price'].iloc[-1] - test_df['price'].iloc[0]
    pct_change = actual_change / test_df['price'].iloc[0] * 100
    
    print(f"\nActual price movement in test period:")
    print(f"  Start: ${test_df['price'].iloc[0]:,.0f}")
    print(f"  End: ${test_df['price'].iloc[-1]:,.0f}")
    print(f"  Change: ${actual_change:,.0f} ({pct_change:+.1f}%)")
    
    # Directional accuracy
    if 'best_trend' in results:
        trend_direction = 'up' if results['best_trend']['slope'] > 0 else 'down'
        actual_direction = 'up' if actual_change > 0 else 'down'
        print(f"\nTrend prediction: {trend_direction}")
        print(f"Actual direction: {actual_direction}")
        print(f"Direction correct: {'Yes' if trend_direction == actual_direction else 'No'}")

def generate_insights(df, results):
    """Generate actionable insights"""
    print("\n" + "="*60)
    print("KEY INSIGHTS & INTERPRETATION")
    print("="*60)
    
    # Current market state
    current_price = df['price'].iloc[-1]
    ma20 = df['price'].rolling(20).mean().iloc[-1]
    ma50 = df['price'].rolling(50).mean().iloc[-1]
    
    print("\n1. CURRENT MARKET STATE")
    print("-" * 30)
    print(f"  Current price: ${current_price:,.0f}")
    print(f"  vs 20-day MA: {(current_price/ma20-1)*100:+.1f}%")
    print(f"  vs 50-day MA: {(current_price/ma50-1)*100:+.1f}%")
    
    # Trend analysis
    print("\n2. TREND ANALYSIS")
    print("-" * 30)
    
    # Short-term trend (10 days)
    short_trend = df['price'].iloc[-10:].pct_change().mean() * 252 * 100
    # Medium-term trend (30 days)
    med_trend = df['price'].iloc[-30:].pct_change().mean() * 252 * 100
    # Long-term trend (90 days)
    long_trend = df['price'].iloc[-90:].pct_change().mean() * 252 * 100
    
    print(f"  10-day trend: {short_trend:+.0f}% annualized")
    print(f"  30-day trend: {med_trend:+.0f}% annualized")
    print(f"  90-day trend: {long_trend:+.0f}% annualized")
    
    # Volatility regime
    print("\n3. VOLATILITY REGIME")
    print("-" * 30)
    
    returns = df['price'].pct_change().dropna()
    current_vol = returns[-20:].std() * np.sqrt(252) * 100
    avg_vol = returns.std() * np.sqrt(252) * 100
    vol_percentile = (returns.rolling(20).std().rank(pct=True).iloc[-1] * 100)
    
    print(f"  Current volatility: {current_vol:.0f}% annualized")
    print(f"  Historical average: {avg_vol:.0f}% annualized")
    print(f"  Percentile: {vol_percentile:.0f}th")
    
    if vol_percentile > 80:
        vol_regime = "HIGH - Expect large price swings"
    elif vol_percentile < 20:
        vol_regime = "LOW - Unusually calm market"
    else:
        vol_regime = "NORMAL - Typical market conditions"
    
    print(f"  Regime: {vol_regime}")
    
    # Model selection advice
    print("\n4. MODEL SELECTION ADVICE")
    print("-" * 30)
    
    if current_vol > avg_vol * 1.5:
        print("  High volatility detected:")
        print("  → Use shorter moving averages (5-10 days)")
        print("  → Volatility bands are more reliable")
        print("  → Trend models may overshoot")
    elif abs(med_trend) > 100:
        print("  Strong trend detected:")
        print("  → Trend extrapolation more reliable")
        print("  → Moving averages may lag")
        print("  → Consider momentum strategies")
    else:
        print("  Range-bound market:")
        print("  → Mean reversion strategies work well")
        print("  → Use longer moving averages")
        print("  → Historical mean is good benchmark")
    
    # Trading signals
    print("\n5. SIMPLE TRADING SIGNALS")
    print("-" * 30)
    
    signals = []
    
    # MA crossover
    if current_price > ma20 > ma50:
        signals.append(("BULLISH", "Price > MA20 > MA50 (Golden alignment)"))
    elif current_price < ma20 < ma50:
        signals.append(("BEARISH", "Price < MA20 < MA50 (Death alignment)"))
    
    # Momentum
    if short_trend > med_trend > long_trend:
        signals.append(("BULLISH", "Accelerating upward momentum"))
    elif short_trend < med_trend < long_trend:
        signals.append(("BEARISH", "Accelerating downward momentum"))
    
    # Volatility
    if vol_percentile > 90:
        signals.append(("CAUTION", "Extreme volatility - reduce position size"))
    
    if signals:
        for signal, reason in signals:
            print(f"  {signal}: {reason}")
    else:
        print("  NEUTRAL: No clear signals")
    
    return {
        'current_state': {
            'price': current_price,
            'ma20': ma20,
            'ma50': ma50,
            'volatility': current_vol,
            'vol_percentile': vol_percentile
        },
        'trends': {
            'short': short_trend,
            'medium': med_trend,
            'long': long_trend
        },
        'signals': signals
    }

def save_results(all_results):
    """Save results to file"""
    output = {
        'test_date': datetime.now().isoformat(),
        'test_type': 'Simple Models Test',
        'results': all_results,
        'data_source': 'Yahoo Finance Cocoa Futures'
    }
    
    from pathlib import Path
    output_file = Path('data/processed/simple_models_test_results.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {output_file}")

def main():
    """Run simple model testing"""
    print("\n" + "#"*60)
    print("# SIMPLE MODEL TESTING WITH REAL DATA")
    print("# Focus on interpretable models that work")
    print("#"*60)
    
    # Load data
    df = load_price_data()
    
    # Test models
    results, train, test = test_simple_models(df)
    
    # Analyze errors
    analyze_errors(results, test)
    
    # Generate insights
    insights = generate_insights(df, results)
    
    # Save everything
    all_results = {
        'model_results': results,
        'insights': insights,
        'data_stats': {
            'total_days': len(df),
            'train_days': len(train),
            'test_days': len(test),
            'price_range': [float(df['price'].min()), float(df['price'].max())]
        }
    }
    
    save_results(all_results)
    
    print("\n" + "#"*60)
    print("# TESTING COMPLETE")
    print("#"*60)

if __name__ == "__main__":
    main()