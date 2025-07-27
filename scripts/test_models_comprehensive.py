#!/usr/bin/env python3
"""
Comprehensive Model Testing with Real Data
Tests all models, measures accuracy, and provides detailed interpretations
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
from src.models.statistical_models import StatisticalSignalModels
from src.models.advanced_time_series_models import AdvancedTimeSeriesModels
from pathlib import Path
from src.models.zen_orchestrator import ZenOrchestrator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_real_price_data():
    """
    Load real price data from database
    """
    print("\nLoading real cocoa price data from database...")
    
    with Session(engine) as session:
        # Get all price data ordered by date
        prices = session.exec(
            select(PriceData)
            .where(PriceData.source == "Yahoo Finance")
            .order_by(PriceData.date)
        ).all()
        
        if not prices:
            print("❌ No price data found in database!")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame([
            {'date': p.date, 'price': p.price}
            for p in prices
        ])
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        print(f"✓ Loaded {len(df)} days of price data")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Price range: ${df['price'].min():,.2f} to ${df['price'].max():,.2f}")
        print(f"  Current price: ${df['price'].iloc[-1]:,.2f}")
        
        return df

def split_train_test(df, test_days=30):
    """
    Split data into train and test sets
    """
    split_date = df.index[-test_days]
    train_df = df[df.index < split_date].copy()
    test_df = df[df.index >= split_date].copy()
    
    print(f"\nData split:")
    print(f"  Training: {len(train_df)} days ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"  Testing: {len(test_df)} days ({test_df.index.min().date()} to {test_df.index.max().date()})")
    
    return train_df, test_df

def test_statistical_models(train_df, test_df):
    """
    Test all statistical models
    """
    print("\n" + "="*80)
    print("TESTING STATISTICAL MODELS")
    print("="*80)
    
    models = StatisticalSignalModels()
    results = {}
    
    # 1. Anomaly Detection
    print("\n1. Anomaly Detection Model")
    try:
        # Prepare data for anomaly detection
        train_data = train_df.copy()
        train_data['returns'] = train_data['price'].pct_change()
        train_data['volatility'] = train_data['returns'].rolling(20).std()
        train_data['volume_proxy'] = train_data['price'].rolling(5).std()  # Proxy for volume
        train_data = train_data.dropna()
        
        anomaly_model = models.build_anomaly_detection_model(train_data[['price', 'returns', 'volatility', 'volume_proxy']])
        
        # Test on test data
        test_data = test_df.copy()
        test_data['returns'] = test_data['price'].pct_change()
        test_data['volatility'] = test_data['returns'].rolling(20).std()
        test_data['volume_proxy'] = test_data['price'].rolling(5).std()
        test_data = test_data.dropna()
        
        if len(test_data) > 0:
            anomalies = anomaly_model['model'].predict(test_data[['price', 'returns', 'volatility', 'volume_proxy']])
            anomaly_days = (anomalies == -1).sum()
            
            results['anomaly_detection'] = {
                'anomaly_days': anomaly_days,
                'anomaly_rate': anomaly_days / len(test_data),
                'feature_importance': anomaly_model.get('feature_importance', {}),
                'interpretation': f"Detected {anomaly_days} anomalous days ({anomaly_days/len(test_data)*100:.1f}% of test period)"
            }
            print(f"  Anomalies detected: {anomaly_days} days")
            print(f"  Anomaly rate: {anomaly_days/len(test_data)*100:.1f}%")
    except Exception as e:
        print(f"  Error: {str(e)}")
        results['anomaly_detection'] = {'error': str(e)}
    
    # 2. Risk Metrics
    print("\n2. Risk Metrics Analysis")
    try:
        train_returns = train_df['price'].pct_change().dropna()
        test_returns = test_df['price'].pct_change().dropna()
        
        risk_metrics = models.calculate_risk_metrics(train_returns)
        
        # Test the VaR predictions
        var_95 = risk_metrics['var_95']
        violations = (test_returns < var_95).sum()
        expected_violations = len(test_returns) * 0.05
        
        results['risk_metrics'] = {
            'var_95': var_95,
            'cvar_95': risk_metrics['cvar_95'],
            'sharpe_ratio': risk_metrics['sharpe_ratio'],
            'violations': violations,
            'expected_violations': expected_violations,
            'interpretation': f"95% VaR: {var_95*100:.1f}%, CVaR: {risk_metrics['cvar_95']*100:.1f}%, Sharpe: {risk_metrics['sharpe_ratio']:.2f}"
        }
        print(f"  95% VaR: {var_95*100:.1f}%")
        print(f"  Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
        print(f"  VaR violations: {violations} (expected {expected_violations:.1f})")
    except Exception as e:
        print(f"  Error: {str(e)}")
        results['risk_metrics'] = {'error': str(e)}
    
    # 3. Granger Causality (if we had multiple series)
    print("\n3. Stationarity Test")
    try:
        stationarity = models.test_stationarity(train_df['price'], "Price")
        results['stationarity'] = {
            'is_stationary': stationarity['is_stationary'],
            'adf_statistic': stationarity['adf_statistic'],
            'p_value': stationarity['p_value'],
            'interpretation': f"Price series is {'stationary' if stationarity['is_stationary'] else 'non-stationary'} (p-value: {stationarity['p_value']:.4f})"
        }
        print(f"  Stationary: {stationarity['is_stationary']}")
        print(f"  ADF p-value: {stationarity['p_value']:.4f}")
    except Exception as e:
        print(f"  Error: {str(e)}")
        results['stationarity'] = {'error': str(e)}
    
    return results

def test_time_series_models(train_df, test_df):
    """
    Test all time series forecasting models
    """
    print("\n" + "="*80)
    print("TESTING TIME SERIES FORECASTING MODELS")
    print("="*80)
    
    models = AdvancedTimeSeriesModels()
    results = {}
    test_days = len(test_df)
    
    # 1. ARIMA
    print("\n1. ARIMA Model")
    try:
        arima_result = models.fit_arima(train_df)
        if arima_result and 'forecast' in arima_result:
            forecast = arima_result['forecast']
            
            # Get forecast values for test period
            if hasattr(forecast, 'forecast'):
                forecast_values = forecast.forecast(steps=test_days)
            else:
                forecast_values = forecast[:test_days]
            
            mae = mean_absolute_error(test_df['price'], forecast_values)
            rmse = np.sqrt(mean_squared_error(test_df['price'], forecast_values))
            mape = np.mean(np.abs((test_df['price'] - forecast_values) / test_df['price'])) * 100
            
            results['arima'] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'order': arima_result.get('order', 'auto'),
                'interpretation': f"ARIMA achieved {mape:.1f}% MAPE with ${mae:.2f} average error. Model order: {arima_result.get('order', 'auto')}"
            }
            print(f"  MAE: ${mae:.2f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAPE: {mape:.1f}%")
    except Exception as e:
        print(f"  Error: {str(e)}")
        results['arima'] = {'error': str(e)}
    
    # 2. SARIMA
    print("\n2. SARIMA Model (Seasonal ARIMA)")
    try:
        sarima_result = models.fit_sarima(train_df)
        if sarima_result and 'forecast' in sarima_result:
            forecast = sarima_result['forecast']
            
            if hasattr(forecast, 'forecast'):
                forecast_values = forecast.forecast(steps=test_days)
            else:
                forecast_values = forecast[:test_days]
            
            mae = mean_absolute_error(test_df['price'], forecast_values)
            rmse = np.sqrt(mean_squared_error(test_df['price'], forecast_values))
            mape = np.mean(np.abs((test_df['price'] - forecast_values) / test_df['price'])) * 100
            
            results['sarima'] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'seasonal_order': sarima_result.get('seasonal_order', 'auto'),
                'interpretation': f"SARIMA achieved {mape:.1f}% MAPE, {'better' if 'arima' in results and mae < results['arima'].get('mae', float('inf')) else 'worse'} than ARIMA"
            }
            print(f"  MAE: ${mae:.2f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAPE: {mape:.1f}%")
    except Exception as e:
        print(f"  Error: {str(e)}")
        results['sarima'] = {'error': str(e)}
    
    # 3. Holt-Winters
    print("\n3. Holt-Winters Exponential Smoothing")
    try:
        hw_result = models.fit_holt_winters(train_df)
        if hw_result and 'forecast' in hw_result:
            forecast_values = hw_result['forecast'][:test_days]
            
            mae = mean_absolute_error(test_df['price'], forecast_values)
            rmse = np.sqrt(mean_squared_error(test_df['price'], forecast_values))
            mape = np.mean(np.abs((test_df['price'] - forecast_values) / test_df['price'])) * 100
            
            results['holt_winters'] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'trend': hw_result.get('trend', 'additive'),
                'seasonal': hw_result.get('seasonal', 'additive'),
                'interpretation': f"Holt-Winters achieved {mape:.1f}% MAPE using {hw_result.get('trend', 'additive')} trend and {hw_result.get('seasonal', 'additive')} seasonality"
            }
            print(f"  MAE: ${mae:.2f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAPE: {mape:.1f}%")
    except Exception as e:
        print(f"  Error: {str(e)}")
        results['holt_winters'] = {'error': str(e)}
    
    # 4. Prophet
    print("\n4. Facebook Prophet")
    try:
        prophet_result = models.fit_prophet(train_df)
        if prophet_result and 'forecast' in prophet_result:
            # Prophet returns a DataFrame with yhat column
            forecast_df = prophet_result['forecast']
            forecast_values = forecast_df['yhat'].iloc[-test_days:].values
            
            mae = mean_absolute_error(test_df['price'], forecast_values)
            rmse = np.sqrt(mean_squared_error(test_df['price'], forecast_values))
            mape = np.mean(np.abs((test_df['price'] - forecast_values) / test_df['price'])) * 100
            
            results['prophet'] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'changepoints': len(prophet_result.get('changepoints', [])),
                'interpretation': f"Prophet achieved {mape:.1f}% MAPE and detected {len(prophet_result.get('changepoints', []))} trend changepoints"
            }
            print(f"  MAE: ${mae:.2f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAPE: {mape:.1f}%")
    except Exception as e:
        print(f"  Error: {str(e)}")
        results['prophet'] = {'error': str(e)}
    
    return results

def test_ml_models(train_df, test_df):
    """
    Test machine learning models
    """
    print("\n" + "="*80)
    print("TESTING MACHINE LEARNING MODELS")
    print("="*80)
    
    models = AdvancedTimeSeriesModels()
    results = {}
    
    # 1. XGBoost
    print("\n1. XGBoost Regressor")
    try:
        xgb_result = models.fit_xgboost(train_df)
        if xgb_result and 'forecast' in xgb_result:
            # XGBoost returns forecast values
            forecast_values = xgb_result['forecast'][:len(test_df)]
            
            mae = mean_absolute_error(test_df['price'], forecast_values)
            rmse = np.sqrt(mean_squared_error(test_df['price'], forecast_values))
            mape = np.mean(np.abs((test_df['price'] - forecast_values) / test_df['price'])) * 100
            
            # Feature importance
            feature_importance = xgb_result.get('feature_importance', {})
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            results['xgboost'] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'top_features': top_features,
                'interpretation': f"XGBoost achieved {mape:.1f}% MAPE. Top features: {', '.join([f[0] for f in top_features]) if top_features else 'N/A'}"
            }
            print(f"  MAE: ${mae:.2f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAPE: {mape:.1f}%")
            if top_features:
                print(f"  Top features: {', '.join([f'{f[0]} ({f[1]:.3f})' for f in top_features])}")
    except Exception as e:
        print(f"  Error: {str(e)}")
        results['xgboost'] = {'error': str(e)}
    
    # 2. LSTM
    print("\n2. LSTM Neural Network")
    try:
        lstm_result = models.fit_lstm_predictor(train_df)
        if lstm_result and 'forecast' in lstm_result:
            # LSTM predictions for test period
            forecast_values = lstm_result['forecast'][:len(test_df)]
            
            mae = mean_absolute_error(test_df['price'], forecast_values)
            rmse = np.sqrt(mean_squared_error(test_df['price'], forecast_values))
            mape = np.mean(np.abs((test_df['price'] - forecast_values) / test_df['price'])) * 100
            
            results['lstm'] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'lookback': lstm_result.get('sequence_length', 30),
                'interpretation': f"LSTM achieved {mape:.1f}% MAPE using {lstm_result.get('sequence_length', 30)}-day lookback window"
            }
            print(f"  MAE: ${mae:.2f}")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAPE: {mape:.1f}%")
    except Exception as e:
        print(f"  Error: {str(e)}")
        results['lstm'] = {'error': str(e)}
    
    return results

def test_volatility_models(train_df, test_df):
    """
    Test volatility and risk models
    """
    print("\n" + "="*80)
    print("TESTING VOLATILITY & RISK MODELS")
    print("="*80)
    
    models = AdvancedTimeSeriesModels()
    results = {}
    
    # 1. GARCH
    print("\n1. GARCH Volatility Model")
    try:
        garch_result = models.fit_garch(train_df)
        if garch_result and 'volatility_forecast' in garch_result:
            # Compare forecasted volatility with realized volatility
            test_returns = test_df['price'].pct_change().dropna()
            realized_vol = test_returns.rolling(5).std() * np.sqrt(252)
            
            vol_forecast = garch_result['volatility_forecast'][:len(realized_vol)]
            vol_mae = mean_absolute_error(realized_vol.dropna(), vol_forecast[:len(realized_vol.dropna())])
            
            results['garch'] = {
                'current_volatility': garch_result.get('current_volatility', 0),
                'volatility_mae': vol_mae,
                'vol_percentile': garch_result.get('volatility_percentile', 0),
                'interpretation': f"GARCH estimates current volatility at {garch_result.get('current_volatility', 0)*100:.1f}% (annualized), which is at the {garch_result.get('volatility_percentile', 0):.0f}th percentile"
            }
            print(f"  Current volatility: {garch_result.get('current_volatility', 0)*100:.1f}%")
            print(f"  Volatility MAE: {vol_mae*100:.1f}%")
    except Exception as e:
        print(f"  Error: {str(e)}")
        results['garch'] = {'error': str(e)}
    
    # 2. Value at Risk
    print("\n2. Value at Risk (VaR)")
    try:
        var_result = models.calculate_value_at_risk(train_df)
        if var_result:
            # Test VaR violations
            test_returns = test_df['price'].pct_change().dropna()
            var_95 = var_result['var_95']
            var_99 = var_result['var_99']
            
            violations_95 = (test_returns < var_95).sum()
            violations_99 = (test_returns < var_99).sum()
            
            expected_95 = len(test_returns) * 0.05
            expected_99 = len(test_returns) * 0.01
            
            results['var'] = {
                'var_95': var_95,
                'var_99': var_99,
                'violations_95': violations_95,
                'violations_99': violations_99,
                'expected_95': expected_95,
                'expected_99': expected_99,
                'interpretation': f"95% VaR is {var_95*100:.1f}% with {violations_95} violations (expected {expected_95:.1f}). Model is {'conservative' if violations_95 < expected_95 else 'aggressive'}"
            }
            print(f"  95% VaR: {var_95*100:.1f}%")
            print(f"  Violations: {violations_95} (expected {expected_95:.1f})")
    except Exception as e:
        print(f"  Error: {str(e)}")
        results['var'] = {'error': str(e)}
    
    return results

def analyze_market_conditions(df):
    """
    Analyze different market conditions in the data
    """
    print("\n" + "="*80)
    print("MARKET CONDITION ANALYSIS")
    print("="*80)
    
    # Calculate returns and volatility
    df['returns'] = df['price'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)
    df['trend'] = df['price'].rolling(50).mean()
    
    # Define market regimes
    conditions = []
    
    # 1. Trending markets
    df['trend_strength'] = (df['price'] - df['trend']) / df['trend']
    strong_uptrend = df[df['trend_strength'] > 0.1]
    strong_downtrend = df[df['trend_strength'] < -0.1]
    
    conditions.append({
        'condition': 'Strong Uptrend',
        'periods': len(strong_uptrend),
        'avg_return': strong_uptrend['returns'].mean() * 252,
        'avg_volatility': strong_uptrend['volatility'].mean(),
        'date_ranges': identify_date_ranges(strong_uptrend)
    })
    
    conditions.append({
        'condition': 'Strong Downtrend',
        'periods': len(strong_downtrend),
        'avg_return': strong_downtrend['returns'].mean() * 252,
        'avg_volatility': strong_downtrend['volatility'].mean(),
        'date_ranges': identify_date_ranges(strong_downtrend)
    })
    
    # 2. High/Low volatility
    high_vol = df[df['volatility'] > df['volatility'].quantile(0.75)]
    low_vol = df[df['volatility'] < df['volatility'].quantile(0.25)]
    
    conditions.append({
        'condition': 'High Volatility',
        'periods': len(high_vol),
        'avg_return': high_vol['returns'].mean() * 252,
        'avg_volatility': high_vol['volatility'].mean(),
        'date_ranges': identify_date_ranges(high_vol)
    })
    
    conditions.append({
        'condition': 'Low Volatility',
        'periods': len(low_vol),
        'avg_return': low_vol['returns'].mean() * 252,
        'avg_volatility': low_vol['volatility'].mean(),
        'date_ranges': identify_date_ranges(low_vol)
    })
    
    # 3. Range-bound markets
    df['price_z'] = (df['price'] - df['price'].rolling(50).mean()) / df['price'].rolling(50).std()
    range_bound = df[df['price_z'].abs() < 1]
    
    conditions.append({
        'condition': 'Range-Bound',
        'periods': len(range_bound),
        'avg_return': range_bound['returns'].mean() * 252,
        'avg_volatility': range_bound['volatility'].mean(),
        'date_ranges': identify_date_ranges(range_bound)
    })
    
    # Print analysis
    for cond in conditions:
        if cond['periods'] > 0:
            print(f"\n{cond['condition']}:")
            print(f"  Periods: {cond['periods']} days")
            print(f"  Avg Annual Return: {cond['avg_return']*100:.1f}%")
            print(f"  Avg Volatility: {cond['avg_volatility']*100:.1f}%")
            print(f"  Recent periods: {', '.join(cond['date_ranges'][:3])}")
    
    return conditions

def identify_date_ranges(df):
    """
    Identify contiguous date ranges
    """
    if len(df) == 0:
        return []
        
    ranges = []
    start_date = df.index[0]
    prev_date = df.index[0]
    
    for date in df.index[1:]:
        if (date - prev_date).days > 5:  # Gap of more than 5 days
            ranges.append(f"{start_date.date()} to {prev_date.date()}")
            start_date = date
        prev_date = date
    
    ranges.append(f"{start_date.date()} to {prev_date.date()}")
    return ranges

def generate_comprehensive_report(all_results, market_conditions):
    """
    Generate comprehensive performance report
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE REPORT")
    print("="*80)
    
    # 1. Overall best performers
    print("\n1. OVERALL BEST PERFORMING MODELS")
    print("-" * 40)
    
    # Collect all forecasting models with MAPE
    forecast_models = []
    for category in ['time_series', 'ml']:
        if category in all_results:
            for model, results in all_results[category].items():
                if 'mape' in results:
                    forecast_models.append((model, results['mape']))
    
    # Sort by MAPE
    forecast_models.sort(key=lambda x: x[1])
    
    print("\nForecasting Accuracy (by MAPE):")
    for i, (model, mape) in enumerate(forecast_models[:5]):
        print(f"  {i+1}. {model.upper()}: {mape:.1f}% error")
    
    # 2. Model recommendations by market condition
    print("\n2. RECOMMENDED MODELS BY MARKET CONDITION")
    print("-" * 40)
    
    recommendations = {
        'Strong Uptrend': {
            'best_model': 'LSTM or XGBoost',
            'reason': 'ML models capture momentum better in trending markets'
        },
        'Strong Downtrend': {
            'best_model': 'GARCH + VaR',
            'reason': 'Risk models crucial during market declines'
        },
        'High Volatility': {
            'best_model': 'GARCH + Bollinger Bands',
            'reason': 'Volatility models excel in turbulent conditions'
        },
        'Low Volatility': {
            'best_model': 'ARIMA or Holt-Winters',
            'reason': 'Traditional time series work well in stable conditions'
        },
        'Range-Bound': {
            'best_model': 'RSI + Bollinger Bands',
            'reason': 'Mean reversion indicators excel in ranging markets'
        }
    }
    
    for condition, rec in recommendations.items():
        print(f"\n{condition}:")
        print(f"  Recommended: {rec['best_model']}")
        print(f"  Reason: {rec['reason']}")
    
    # 3. Key insights
    print("\n3. KEY INSIGHTS & INTERPRETATIONS")
    print("-" * 40)
    
    insights = []
    
    # Check if ML models outperform traditional
    if 'ml' in all_results and 'time_series' in all_results:
        ml_avg = np.mean([r.get('mape', 100) for r in all_results['ml'].values() if 'mape' in r])
        ts_avg = np.mean([r.get('mape', 100) for r in all_results['time_series'].values() if 'mape' in r])
        
        if ml_avg < ts_avg:
            insights.append(f"Machine Learning models ({ml_avg:.1f}% avg MAPE) outperformed traditional time series ({ts_avg:.1f}% avg MAPE) by {ts_avg - ml_avg:.1f}%")
        else:
            insights.append(f"Traditional time series models ({ts_avg:.1f}% avg MAPE) performed better than ML models ({ml_avg:.1f}% avg MAPE)")
    
    # Technical indicators insights
    if 'statistical' in all_results:
        stat_results = all_results['statistical']
        if 'rsi' in stat_results and stat_results['rsi'].get('current_value'):
            rsi_val = stat_results['rsi']['current_value']
            if rsi_val > 70:
                insights.append(f"RSI at {rsi_val:.1f} indicates overbought conditions - potential reversal ahead")
            elif rsi_val < 30:
                insights.append(f"RSI at {rsi_val:.1f} indicates oversold conditions - potential bounce ahead")
    
    # Volatility insights
    if 'volatility' in all_results and 'garch' in all_results['volatility']:
        vol_percentile = all_results['volatility']['garch'].get('vol_percentile', 50)
        if vol_percentile > 80:
            insights.append(f"Volatility in top {100-vol_percentile:.0f}% historically - expect large price swings")
        elif vol_percentile < 20:
            insights.append(f"Volatility in bottom {vol_percentile:.0f}% historically - unusually calm market")
    
    for i, insight in enumerate(insights):
        print(f"\n{i+1}. {insight}")
    
    # 4. Trading recommendations
    print("\n4. TRADING RECOMMENDATIONS")
    print("-" * 40)
    
    # Synthesize signals from multiple models
    signals = []
    
    # Check technical indicators
    if 'statistical' in all_results:
        stat = all_results['statistical']
        if 'macd' in stat and stat['macd'].get('trend') == 'bullish':
            signals.append(('bullish', 'MACD shows bullish trend'))
        if 'rsi' in stat and stat['rsi'].get('current_value', 50) < 30:
            signals.append(('bullish', 'RSI oversold'))
        if 'bollinger_bands' in stat and stat['bollinger_bands'].get('current_position') == 'below lower':
            signals.append(('bullish', 'Price below lower Bollinger Band'))
    
    # Count signals
    bullish_count = sum(1 for s in signals if s[0] == 'bullish')
    bearish_count = sum(1 for s in signals if s[0] == 'bearish')
    
    if bullish_count > bearish_count + 1:
        print("\nOVERALL SIGNAL: BULLISH")
        print("Reasons:")
        for signal, reason in signals:
            if signal == 'bullish':
                print(f"  - {reason}")
    elif bearish_count > bullish_count + 1:
        print("\nOVERALL SIGNAL: BEARISH")
        print("Reasons:")
        for signal, reason in signals:
            if signal == 'bearish':
                print(f"  - {reason}")
    else:
        print("\nOVERALL SIGNAL: NEUTRAL")
        print("Mixed signals - wait for clearer direction")
    
    return {
        'best_models': forecast_models[:3],
        'recommendations': recommendations,
        'insights': insights,
        'signal_count': {'bullish': bullish_count, 'bearish': bearish_count}
    }

def save_results(all_results, report):
    """
    Save comprehensive results to file
    """
    output = {
        'test_date': datetime.now().isoformat(),
        'model_results': all_results,
        'report': report,
        'data_source': 'Real Yahoo Finance cocoa futures data'
    }
    
    output_file = Path('data/processed/comprehensive_model_test_results.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {output_file}")

def main():
    """
    Run comprehensive model testing
    """
    print("\n" + "#"*80)
    print("# COMPREHENSIVE MODEL TESTING WITH REAL COCOA DATA")
    print("# Testing all models to determine which perform best")
    print("#"*80)
    
    # Load data
    df = load_real_price_data()
    if df is None:
        return
    
    # Split data
    train_df, test_df = split_train_test(df, test_days=30)
    
    # Test all model categories
    all_results = {
        'statistical': test_statistical_models(train_df, test_df),
        'time_series': test_time_series_models(train_df, test_df),
        'ml': test_ml_models(train_df, test_df),
        'volatility': test_volatility_models(train_df, test_df)
    }
    
    # Analyze market conditions
    market_conditions = analyze_market_conditions(df)
    
    # Generate report
    report = generate_comprehensive_report(all_results, market_conditions)
    
    # Save results
    save_results(all_results, report)
    
    print("\n" + "#"*80)
    print("# TESTING COMPLETE")
    print("#"*80)

if __name__ == "__main__":
    main()