"""
Test ALL v2 Time Series Models with v3 Real Data
NO FAKE DATA - 100% REAL INCLUDING UN COMTRADE
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our advanced models
from src.models.advanced_time_series_models import AdvancedTimeSeriesModels

print("="*80)
print("TESTING ALL V2 TIME SERIES MODELS WITH V3 REAL DATA")
print("="*80)

# Load real unified data
print("\n1. LOADING REAL UNIFIED DATA")
print("-" * 60)
df = pd.read_csv("data/processed/unified_real_data.csv")
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
df.set_index('date', inplace=True)
print(f"✓ Loaded {len(df)} days of unified REAL data")
print(f"✓ Date range: {df.index.min().date()} to {df.index.max().date()}")
print(f"✓ Export concentration from REAL UN Comtrade: mean={df['export_concentration'].mean():.3f}")

# Initialize models
models = AdvancedTimeSeriesModels()

# Run all models
print("\n2. RUNNING ALL TIME SERIES MODELS")
print("-" * 60)

all_results = models.run_all_models(df)

# Summarize results
print("\n" + "="*80)
print("COMPREHENSIVE MODEL RESULTS SUMMARY")
print("="*80)

# Traditional Time Series
print("\n1. TRADITIONAL TIME SERIES MODELS:")
print("-" * 40)

if 'arima' in all_results:
    arima = all_results['arima']
    print(f"ARIMA{arima['order']}:")
    print(f"  - RMSE: ${arima['rmse']:.2f}")
    print(f"  - MAPE: {arima['mape']:.2f}%")
    print(f"  - AIC: {arima['aic']:.2f}")

if 'var' in all_results:
    var = all_results['var']
    print(f"\nVAR (lag={var['lag_order']}):")
    print(f"  Granger Causality (→ price):")
    for var_name, result in var['causality'].items():
        if result['causes_price']:
            print(f"    - {var_name}: p={result['p_value']:.4f} ***")

if 'prophet' in all_results:
    prophet = all_results['prophet']
    print(f"\nProphet:")
    print(f"  - RMSE: ${prophet['rmse']:.2f}")
    print(f"  - MAPE: {prophet['mape']:.2f}%")
    print(f"  - Changepoints: {len(prophet['changepoints'])}")

# Volatility Models
print("\n2. VOLATILITY MODELS:")
print("-" * 40)

if 'garch' in all_results:
    garch = all_results['garch']
    print(f"GARCH:")
    print(f"  - Current volatility: {garch['conditional_volatility'].iloc[-1]:.2f}%")
    print(f"  - 95% VaR: {garch['var_95']:.2f}%")
    print(f"  - High vol periods: {len(garch['high_vol_periods'])}")

if 'ewma' in all_results:
    ewma = all_results['ewma']
    print(f"\nEWMA:")
    print(f"  - Current volatility: {ewma['current_vol']:.2f}%")
    print(f"  - Volatility percentile: {ewma['vol_percentile']:.1f}%")

# ML Models
print("\n3. MACHINE LEARNING MODELS:")
print("-" * 40)

if 'xgboost' in all_results and all_results['xgboost']:
    xgb_res = all_results['xgboost']
    print(f"XGBoost ({xgb_res['target_days']}-day forecast):")
    print(f"  - CV MAPE: {np.mean(xgb_res['cv_scores']):.2f}%")
    print(f"  - Top features: {', '.join(xgb_res['feature_importance'].head(3)['feature'].tolist())}")

if 'lstm_autoencoder' in all_results:
    lstm_ae = all_results['lstm_autoencoder']
    print(f"\nLSTM Autoencoder:")
    print(f"  - Anomalies: {len(lstm_ae['anomaly_dates'])} detected")
    print(f"  - Threshold: {lstm_ae['anomaly_threshold']:.4f}")

if 'lstm_predictor' in all_results:
    lstm_pred = all_results['lstm_predictor']
    print(f"\nLSTM Predictor:")
    print(f"  - Test RMSE: ${lstm_pred['test_rmse']:.2f}")
    print(f"  - Test MAPE: {lstm_pred['test_mape']:.2f}%")

# Statistical Process Control
print("\n4. STATISTICAL PROCESS CONTROL:")
print("-" * 40)

if 'cusum' in all_results:
    cusum = all_results['cusum']
    print(f"CUSUM:")
    print(f"  - Change points: {len(cusum['change_points'])}")
    
if 'modified_zscore' in all_results:
    zscore = all_results['modified_zscore']
    print(f"\nModified Z-score:")
    print(f"  - Outliers: {len(zscore['outliers'])}")
    
if 'lof' in all_results:
    lof = all_results['lof']
    print(f"\nLocal Outlier Factor:")
    print(f"  - Anomalies: {len(lof['anomaly_dates'])}")

# Identify key events detected
print("\n" + "="*80)
print("KEY EVENTS DETECTED BY MODELS")
print("="*80)

# Check if models detected the major price surge
surge_period = df['2024-02-01':'2024-04-30']
surge_start = surge_period['price'].iloc[0]
surge_peak = surge_period['price'].max()
surge_increase = (surge_peak - surge_start) / surge_start * 100

print(f"\nActual surge: ${surge_start:,.0f} → ${surge_peak:,.0f} ({surge_increase:.1f}% increase)")
print("\nModel detection performance:")

# Check each anomaly detection method
detection_summary = []

if 'cusum' in all_results:
    cusum_detections = [cp for cp in all_results['cusum']['change_points'] 
                       if '2024-01' <= str(cp) <= '2024-03']
    if cusum_detections:
        detection_summary.append(f"✓ CUSUM: Detected {len(cusum_detections)} change points")
    else:
        detection_summary.append("✗ CUSUM: No detection")

if 'modified_zscore' in all_results:
    zscore_detections = [date for date in all_results['modified_zscore']['outliers'].index
                        if '2024-02' <= str(date) <= '2024-04']
    if zscore_detections:
        detection_summary.append(f"✓ Z-score: Detected {len(zscore_detections)} outliers")
    else:
        detection_summary.append("✗ Z-score: No detection")

if 'lof' in all_results:
    lof_detections = [date for date in all_results['lof']['anomaly_dates']
                     if '2024-01' <= str(date) <= '2024-04']
    if lof_detections:
        detection_summary.append(f"✓ LOF: Detected {len(lof_detections)} anomalies")
    else:
        detection_summary.append("✗ LOF: No detection")

if 'lstm_autoencoder' in all_results:
    lstm_detections = [date for date in all_results['lstm_autoencoder']['anomaly_dates']
                      if '2024-01' <= str(date) <= '2024-04']
    if lstm_detections:
        detection_summary.append(f"✓ LSTM: Detected {len(lstm_detections)} anomalies")
    else:
        detection_summary.append("✗ LSTM: No detection")

for summary in detection_summary:
    print(f"  {summary}")

# Model recommendations based on market conditions
print("\n" + "="*80)
print("MODEL RECOMMENDATIONS BY MARKET REGIME")
print("="*80)

# Check current volatility regime
if 'garch' in all_results and 'ewma' in all_results:
    current_vol = all_results['ewma']['current_vol']
    vol_percentile = all_results['ewma']['vol_percentile']
    
    # Handle NaN percentile
    if np.isnan(vol_percentile):
        # Use GARCH volatility levels instead
        garch_vol = all_results['garch']['conditional_volatility'].iloc[-1]
        high_vol_threshold = all_results['garch']['conditional_volatility'].quantile(0.8)
        if garch_vol > high_vol_threshold:
            vol_percentile = 85
        else:
            vol_percentile = 50
    
    print(f"\nCurrent market volatility: {current_vol:.1f}% (percentile: {vol_percentile:.0f}%)")
    
    if vol_percentile > 80:
        print("\n🔴 HIGH VOLATILITY REGIME")
        print("Recommended models:")
        print("  1. GARCH - For volatility clustering")
        print("  2. LSTM - For capturing non-linear patterns")
        print("  3. EWMA - For adaptive volatility tracking")
        print("  4. Modified Z-score - For robust outlier detection")
    elif vol_percentile > 50:
        print("\n🟡 MODERATE VOLATILITY REGIME")
        print("Recommended models:")
        print("  1. VAR - For multivariate relationships")
        print("  2. XGBoost - For feature-rich predictions")
        print("  3. Prophet - For trend/seasonality")
        print("  4. CUSUM - For change detection")
    else:
        print("\n🟢 LOW VOLATILITY REGIME")
        print("Recommended models:")
        print("  1. ARIMA - For stable time series")
        print("  2. Prophet - For trend decomposition")
        print("  3. VAR - For lead-lag relationships")
        print("  4. LOF - For subtle anomalies")

# Save comprehensive results
results_summary = {
    "test_date": datetime.now().isoformat(),
    "data_validation": {
        "total_days": len(df),
        "date_range": f"{df.index.min().date()} to {df.index.max().date()}",
        "real_export_concentration": {
            "mean": float(df['export_concentration'].mean()),
            "std": float(df['export_concentration'].std()),
            "min": float(df['export_concentration'].min()),
            "max": float(df['export_concentration'].max())
        }
    },
    "models_tested": list(all_results.keys()),
    "surge_detection": {
        "actual_surge": f"{surge_increase:.1f}%",
        "detection_summary": detection_summary
    },
    "current_regime": {
        "volatility": float(current_vol) if 'current_vol' in locals() else None,
        "percentile": float(vol_percentile) if 'vol_percentile' in locals() else None
    }
}

# Save results
with open("data/processed/v2_models_test_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print("\n✓ Results saved to data/processed/v2_models_test_results.json")
print("\n" + "="*80)
print("ALL V2 MODELS TESTED WITH V3 REAL DATA")
print("Ready for orchestration implementation!")
print("="*80)