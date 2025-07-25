"""
Test V2 Models with Real Data

Tests all the comprehensive models from v2 including:
- Granger Causality
- Random Forest
- Isolation Forest
- Regime Detection
- Time Series Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our models
from src.models.statistical_models import StatisticalSignalModels

def prepare_real_data():
    """Prepare our real data for v2 models"""
    print("Loading and preparing real data...")
    
    # Load price data
    price_path = Path("data/historical/prices/cocoa_daily_prices_2yr.csv")
    prices_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
    
    # Convert to price dict format expected by v2 models
    price_data = {
        date.strftime('%Y-%m-%d'): price 
        for date, price in prices_df['cocoa_cc_close'].items()
    }
    
    # Load weather data
    weather_path = Path("data/historical/weather/all_locations_weather_2yr.csv")
    weather_df = pd.read_csv(weather_path, index_col=0, parse_dates=True)
    
    # Calculate weather anomalies (deviation from mean)
    weather_data = {}
    for date in weather_df.index.unique():
        day_data = weather_df.loc[date]
        if isinstance(day_data, pd.Series):
            day_data = day_data.to_frame().T
        
        # Calculate anomalies
        avg_temp = day_data['temp_mean_c'].mean() if 'temp_mean_c' in day_data else 26.5
        avg_rain = day_data['precipitation_mm'].mean() if 'precipitation_mm' in day_data else 3.5
        
        weather_data[date.strftime('%Y-%m-%d')] = {
            'avg_rainfall_anomaly': (avg_rain - 3.5) / 3.5,  # Normalized anomaly
            'avg_temp_anomaly': (avg_temp - 26.5) / 26.5
        }
    
    # Create synthetic trade data based on price movements
    # (Since we don't have real trade volume data yet)
    trade_data = {}
    for i, (date, price) in enumerate(prices_df['cocoa_cc_close'].items()):
        if i > 0:
            prev_price = prices_df['cocoa_cc_close'].iloc[i-1]
            volume_change = (price - prev_price) / prev_price * 100
        else:
            volume_change = 0
            
        trade_data[date.strftime('%Y-%m-%d')] = {
            'volume_change_pct': volume_change,
            'export_concentration': 0.65 + np.random.normal(0, 0.05)  # ~65% concentration
        }
    
    print(f"✓ Prepared {len(price_data)} days of price data")
    print(f"✓ Prepared {len(weather_data)} days of weather data")
    print(f"✓ Prepared {len(trade_data)} days of trade data")
    
    return weather_data, trade_data, price_data

def test_all_v2_models():
    """Run all v2 models with real data"""
    
    # Initialize models
    models = StatisticalSignalModels()
    
    # Prepare data
    weather_data, trade_data, price_data = prepare_real_data()
    
    # Prepare time series data
    df = models.prepare_time_series_data(weather_data, trade_data, price_data)
    print(f"\nPrepared time series data: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    results = {}
    
    # 1. Test Stationarity
    print("\n" + "="*60)
    print("1. STATIONARITY TESTS")
    print("="*60)
    for col in df.columns:
        adf_stat, p_value, critical_values = models.test_stationarity(df[col], name=col)
        print(f"\n{col}:")
        print(f"  ADF Statistic: {adf_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Stationary: {'Yes' if p_value < 0.05 else 'No'}")
    
    # 2. Granger Causality Tests
    print("\n" + "="*60)
    print("2. GRANGER CAUSALITY TESTS")
    print("="*60)
    
    test_cols = ['rainfall_anomaly', 'temperature_anomaly', 'trade_volume_change']
    granger_results = models.granger_causality_test(df, 'price', test_cols, max_lag=7)
    results['granger_causality'] = granger_results
    
    # 3. Anomaly Detection with Isolation Forest
    print("\n" + "="*60)
    print("3. ANOMALY DETECTION (ISOLATION FOREST)")
    print("="*60)
    
    anomaly_results = models.build_anomaly_detection_model(df, contamination=0.05)
    
    # Find top anomalies
    anomaly_df = df.copy()
    anomaly_df['anomaly'] = anomaly_results['predictions']
    anomaly_df['anomaly_score'] = anomaly_results['scores']
    
    anomalies = anomaly_df[anomaly_df['anomaly'] == -1].sort_values('anomaly_score')
    print(f"\nDetected {len(anomalies)} anomalies out of {len(df)} days")
    print("\nTop 10 Anomalous Days:")
    for date, row in anomalies.head(10).iterrows():
        print(f"  {date.date()}: Price=${row['price']:,.0f}, Score={row['anomaly_score']:.3f}")
    
    results['anomaly_detection'] = {
        'n_anomalies': len(anomalies),
        'contamination': 0.05,
        'feature_importance': anomaly_results['feature_importance']
    }
    
    # 4. Predictive Model (Random Forest)
    print("\n" + "="*60)
    print("4. PREDICTIVE MODEL (RANDOM FOREST)")
    print("="*60)
    
    rf_results = models.build_predictive_model(df, target='price', test_size=0.2)
    
    print(f"\nRandom Forest Performance:")
    print(f"  Train R²: {rf_results['train_metrics']['r2']:.3f}")
    print(f"  Test R²: {rf_results['test_metrics']['r2']:.3f}")
    print(f"  Test RMSE: ${rf_results['test_metrics']['rmse']:.2f}")
    print(f"  Test MAPE: {rf_results['test_metrics']['mape']:.1f}%")
    
    print("\nFeature Importance:")
    for feature, importance in rf_results['feature_importance'].items():
        print(f"  {feature}: {importance:.3f}")
    
    results['random_forest'] = rf_results
    
    # 5. Regime Detection
    print("\n" + "="*60)
    print("5. REGIME DETECTION")
    print("="*60)
    
    price_series = df['price']
    regime_results = models.perform_regime_detection(price_series, n_regimes=3)
    
    # Analyze regimes
    regime_df = pd.DataFrame({
        'price': price_series,
        'regime': regime_results['regimes'],
        'returns': price_series.pct_change()
    })
    
    print(f"\nDetected {regime_results['n_regimes']} regimes")
    for regime in range(regime_results['n_regimes']):
        regime_data = regime_df[regime_df['regime'] == regime]
        if len(regime_data) > 0:
            print(f"\nRegime {regime}:")
            print(f"  Days: {len(regime_data)}")
            print(f"  Avg Price: ${regime_data['price'].mean():,.0f}")
            print(f"  Avg Daily Return: {regime_data['returns'].mean()*100:.2f}%")
            print(f"  Volatility: {regime_data['returns'].std()*100:.2f}%")
            print(f"  Date Range: {regime_data.index[0].date()} to {regime_data.index[-1].date()}")
    
    results['regime_detection'] = regime_results
    
    # 6. Signal Correlations
    print("\n" + "="*60)
    print("6. SIGNAL CORRELATIONS")
    print("="*60)
    
    correlation_results = models.calculate_signal_correlations(df)
    results['correlations'] = correlation_results
    
    # 7. Risk Metrics
    print("\n" + "="*60)
    print("7. RISK METRICS")
    print("="*60)
    
    returns = df['price'].pct_change().dropna()
    risk_metrics = models.calculate_risk_metrics(returns)
    
    print(f"  Daily VaR (95%): {risk_metrics['daily_var']*100:.2f}%")
    print(f"  Daily CVaR (95%): {risk_metrics['daily_cvar']*100:.2f}%")
    print(f"  Annualized Volatility: {risk_metrics['annualized_volatility']*100:.1f}%")
    print(f"  Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}")
    print(f"  Max Drawdown: {risk_metrics['max_drawdown']*100:.1f}%")
    
    results['risk_metrics'] = risk_metrics
    
    # Save comprehensive results
    report_path = Path("data/processed/v2_models_test_report.json")
    report_path.parent.mkdir(exist_ok=True)
    
    report = {
        "test_date": datetime.now().isoformat(),
        "data_period": f"{df.index[0]} to {df.index[-1]}",
        "n_observations": len(df),
        "models_tested": [
            "Stationarity Tests (ADF)",
            "Granger Causality",
            "Isolation Forest (Anomaly Detection)",
            "Random Forest (Price Prediction)",
            "Regime Detection",
            "Signal Correlations",
            "Risk Metrics"
        ],
        "results": results
    }
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=convert_numpy)
    
    print(f"\n✓ Comprehensive test report saved to {report_path}")
    
    return results

if __name__ == "__main__":
    print("="*60)
    print("V2 MODELS COMPREHENSIVE TEST")
    print("Testing all time series and ML models with real data")
    print("="*60)
    
    results = test_all_v2_models()
    
    print("\n" + "="*60)
    print("✓ ALL V2 MODEL TESTS COMPLETED SUCCESSFULLY")
    print("="*60)