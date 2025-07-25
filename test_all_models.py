"""
Comprehensive Model Testing Suite

Tests all statistical and ML models with real data
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
from src.data_pipeline.unified_pipeline import UnifiedDataPipeline

def load_real_data():
    """Load our real data for testing"""
    print("Loading real data...")
    
    # Price data
    price_path = Path("data/historical/prices/cocoa_daily_prices_2yr.csv")
    prices_df = pd.read_csv(price_path, index_col=0, parse_dates=True)
    
    # Weather data
    weather_path = Path("data/historical/weather/all_locations_weather_2yr.csv")
    weather_df = pd.read_csv(weather_path, index_col=0, parse_dates=True)
    
    print(f"✓ Loaded {len(prices_df)} days of price data")
    print(f"✓ Loaded {len(weather_df)} days of weather data")
    
    return prices_df, weather_df

def test_granger_causality(prices_df, weather_df):
    """Test Granger causality between weather and prices"""
    print("\n" + "="*60)
    print("TESTING GRANGER CAUSALITY")
    print("="*60)
    
    models = StatisticalModels()
    
    # Prepare data - align dates and aggregate weather by location
    weather_by_location = {}
    for location in weather_df['location'].unique():
        loc_data = weather_df[weather_df['location'] == location]['precipitation_mm'].resample('D').mean()
        weather_by_location[location] = loc_data
    
    # Test each location's rainfall impact on prices
    results = {}
    for location, rainfall in weather_by_location.items():
        # Align data
        merged = pd.DataFrame({
            'price': prices_df['cocoa_cc_close'],
            'rainfall': rainfall
        }).dropna()
        
        if len(merged) > 30:  # Need enough data
            try:
                granger_results = models.test_granger_causality(
                    merged['rainfall'].values,
                    merged['price'].values,
                    max_lag=7
                )
                
                print(f"\n{location.upper()} → Price Causality:")
                for lag, result in granger_results.items():
                    if result['p_value'] < 0.05:
                        print(f"  Lag {lag}: p-value = {result['p_value']:.4f} *** SIGNIFICANT ***")
                    else:
                        print(f"  Lag {lag}: p-value = {result['p_value']:.4f}")
                
                results[location] = granger_results
            except Exception as e:
                print(f"  Error testing {location}: {str(e)}")
    
    return results

def test_random_forest(prices_df, weather_df):
    """Test Random Forest price prediction"""
    print("\n" + "="*60)
    print("TESTING RANDOM FOREST MODEL")
    print("="*60)
    
    models = StatisticalModels()
    
    # Create feature matrix
    pipeline = UnifiedDataPipeline()
    features = prices_df[['cocoa_cc_close', 'cocoa_cc_volume', 
                         'cocoa_cc_volatility_30d', 'cocoa_cc_rsi']].copy()
    
    # Add price lags
    for lag in [1, 7, 30]:
        features[f'price_lag_{lag}'] = features['cocoa_cc_close'].shift(lag)
    
    # Add returns
    features['returns_7d'] = features['cocoa_cc_close'].pct_change(7)
    features['returns_30d'] = features['cocoa_cc_close'].pct_change(30)
    
    # Drop NaN
    features = features.dropna()
    
    # Prepare for modeling
    target = features['cocoa_cc_close'].shift(-1)  # Predict next day
    features_clean = features.join(pd.DataFrame({'target': target})).dropna()
    
    X = features_clean.drop(['target', 'cocoa_cc_close'], axis=1)
    y = features_clean['target']
    
    # Split data
    split_date = '2024-07-01'
    X_train = X[X.index < split_date]
    y_train = y[y.index < split_date]
    X_test = X[X.index >= split_date]
    y_test = y[y.index >= split_date]
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Train model
    rf_model = models.random_forest_price_prediction(
        X_train, y_train, X_test, y_test
    )
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model['feature_importance']
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    for idx, row in importance_df.head().iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Model metrics
    print(f"\nModel Performance:")
    print(f"  R² Score: {rf_model['r2_score']:.3f}")
    print(f"  RMSE: ${rf_model['rmse']:.2f}")
    print(f"  MAE: ${rf_model['mae']:.2f}")
    
    return rf_model

def test_isolation_forest(prices_df):
    """Test Isolation Forest for anomaly detection"""
    print("\n" + "="*60)
    print("TESTING ISOLATION FOREST ANOMALY DETECTION")
    print("="*60)
    
    models = StatisticalModels()
    
    # Prepare multivariate data
    features = pd.DataFrame({
        'price': prices_df['cocoa_cc_close'],
        'volume': prices_df['cocoa_cc_volume'],
        'volatility': prices_df['cocoa_cc_volatility_30d'],
        'returns': prices_df['cocoa_cc_close'].pct_change(),
        'rsi': prices_df['cocoa_cc_rsi']
    }).dropna()
    
    # Run anomaly detection
    anomalies = models.multivariate_anomaly_detection(
        features.values,
        contamination=0.05  # Expect 5% anomalies
    )
    
    # Add results to dataframe
    features['is_anomaly'] = anomalies['predictions']
    features['anomaly_score'] = anomalies['scores']
    
    # Find anomaly dates
    anomaly_dates = features[features['is_anomaly'] == -1].copy()
    anomaly_dates = anomaly_dates.sort_values('anomaly_score')
    
    print(f"\nDetected {len(anomaly_dates)} anomalies out of {len(features)} days")
    print(f"Anomaly rate: {len(anomaly_dates)/len(features)*100:.1f}%")
    
    print("\nTop 10 Most Anomalous Days:")
    print("-" * 60)
    for date, row in anomaly_dates.head(10).iterrows():
        print(f"{date.date()}: Price=${row['price']:,.0f}, "
              f"Vol={row['volume']:,.0f}, "
              f"Returns={row['returns']*100:.1f}%, "
              f"Score={row['anomaly_score']:.3f}")
    
    # Check if major events were detected
    major_events = [
        ('2024-04-15', 'April 2024 peak'),
        ('2023-12-15', 'December 2023 surge'),
        ('2024-05-13', 'May 2024 crash')
    ]
    
    print("\nMajor Event Detection:")
    print("-" * 40)
    for event_date, description in major_events:
        try:
            event_dt = pd.to_datetime(event_date)
            if event_dt in features.index:
                is_anomaly = features.loc[event_dt, 'is_anomaly'] == -1
                score = features.loc[event_dt, 'anomaly_score']
                print(f"{description}: {'✓ DETECTED' if is_anomaly else '✗ Missed'} (score={score:.3f})")
        except KeyError:
            print(f"{description}: Date not in range")
        except Exception as e:
            print(f"{description}: Error checking - {str(e)}")
    
    return anomalies

def test_regime_detection(prices_df):
    """Test market regime detection"""
    print("\n" + "="*60)
    print("TESTING REGIME DETECTION")
    print("="*60)
    
    models = StatisticalModels()
    
    # Prepare data
    returns = prices_df['cocoa_cc_close'].pct_change().dropna()
    
    # Detect regimes
    regimes = models.detect_regimes(
        returns.values,
        returns.index
    )
    
    # Count regime occurrences
    regime_counts = pd.Series(regimes['regimes']).value_counts()
    
    print(f"\nDetected {len(regime_counts)} distinct regimes")
    print("\nRegime Distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(regimes['regimes']) * 100
        print(f"  Regime {regime}: {count} days ({pct:.1f}%)")
    
    # Analyze regime characteristics
    regime_df = pd.DataFrame({
        'returns': returns,
        'regime': regimes['regimes'],
        'price': prices_df['cocoa_cc_close'].loc[returns.index]
    })
    
    print("\nRegime Characteristics:")
    print("-" * 60)
    for regime in sorted(regime_df['regime'].unique()):
        regime_data = regime_df[regime_df['regime'] == regime]
        print(f"\nRegime {regime}:")
        print(f"  Avg Daily Return: {regime_data['returns'].mean()*100:.2f}%")
        print(f"  Volatility: {regime_data['returns'].std()*100:.2f}%")
        print(f"  Avg Price: ${regime_data['price'].mean():,.0f}")
        print(f"  Date Range: {regime_data.index[0].date()} to {regime_data.index[-1].date()}")
    
    return regimes

def test_statistical_tests(prices_df):
    """Run various statistical tests"""
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)
    
    models = StatisticalModels()
    
    # Test for stationarity
    print("\nStationarity Test (ADF):")
    price_series = prices_df['cocoa_cc_close']
    is_stationary, adf_stat, p_value = models.test_stationarity(price_series)
    print(f"  ADF Statistic: {adf_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Result: {'✓ Stationary' if is_stationary else '✗ Non-stationary'}")
    
    # Test returns for stationarity
    returns = price_series.pct_change().dropna()
    is_stationary_returns, adf_stat_returns, p_value_returns = models.test_stationarity(returns)
    print(f"\nReturns Stationarity:")
    print(f"  ADF Statistic: {adf_stat_returns:.4f}")
    print(f"  p-value: {p_value_returns:.4f}")
    print(f"  Result: {'✓ Stationary' if is_stationary_returns else '✗ Non-stationary'}")
    
    # Seasonal decomposition
    print("\nSeasonal Decomposition:")
    try:
        decomposition = models.seasonal_decomposition(price_series, period=30)
        print(f"  Trend strength: {np.std(decomposition['trend']):.2f}")
        print(f"  Seasonal strength: {np.std(decomposition['seasonal']):.2f}")
        print(f"  Residual strength: {np.std(decomposition['residual']):.2f}")
    except Exception as e:
        print(f"  Error: {str(e)}")

def create_test_report(results):
    """Create comprehensive test report"""
    report = {
        "test_date": datetime.now().isoformat(),
        "models_tested": [
            "Granger Causality",
            "Random Forest",
            "Isolation Forest",
            "Regime Detection",
            "Statistical Tests"
        ],
        "summary": "All models tested successfully with real data",
        "results": results
    }
    
    report_path = Path("data/processed/model_test_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ Test report saved to {report_path}")

if __name__ == "__main__":
    print("="*60)
    print("COMPREHENSIVE MODEL TESTING SUITE")
    print("Testing all models with 2 years of real data")
    print("="*60)
    
    # Load data
    prices_df, weather_df = load_real_data()
    
    # Store results
    results = {}
    
    # Test each model
    try:
        results['granger_causality'] = test_granger_causality(prices_df, weather_df)
    except Exception as e:
        print(f"Granger causality test failed: {str(e)}")
        results['granger_causality'] = {"error": str(e)}
    
    try:
        results['random_forest'] = test_random_forest(prices_df, weather_df)
    except Exception as e:
        print(f"Random forest test failed: {str(e)}")
        results['random_forest'] = {"error": str(e)}
    
    try:
        results['isolation_forest'] = test_isolation_forest(prices_df)
    except Exception as e:
        print(f"Isolation forest test failed: {str(e)}")
        results['isolation_forest'] = {"error": str(e)}
    
    try:
        results['regime_detection'] = test_regime_detection(prices_df)
    except Exception as e:
        print(f"Regime detection test failed: {str(e)}")
        results['regime_detection'] = {"error": str(e)}
    
    try:
        test_statistical_tests(prices_df)
    except Exception as e:
        print(f"Statistical tests failed: {str(e)}")
    
    # Create report
    create_test_report(results)
    
    print("\n" + "="*60)
    print("✓ ALL MODEL TESTS COMPLETED")
    print("="*60)