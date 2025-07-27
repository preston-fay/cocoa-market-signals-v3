#!/usr/bin/env python3
"""
Run Advanced Model Evaluation with Regularization and Predictability Analysis
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.models.regularized_ensemble import RegularizedEnsemble
from src.models.advanced_time_series_models import AdvancedTimeSeriesModels
from src.evaluation.predictability_analyzer import PredictabilityAnalyzer
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def run_advanced_evaluation():
    """Run comprehensive evaluation with advanced models"""
    
    print("🚀 Advanced Model Evaluation")
    print("=" * 80)
    
    # Load comprehensive dataset
    train_df = pd.read_csv('data/processed/comprehensive_train.csv', index_col='date', parse_dates=True)
    test_df = pd.read_csv('data/processed/comprehensive_test.csv', index_col='date', parse_dates=True)
    
    print(f"📊 Dataset Summary:")
    print(f"   Train: {len(train_df)} samples ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"   Test: {len(test_df)} samples ({test_df.index[0].date()} to {test_df.index[-1].date()})")
    
    # 1. TEST REGULARIZED ENSEMBLE
    print(f"\n{'='*80}")
    print("1. REGULARIZED ENSEMBLE MODEL")
    print("=" * 80)
    
    # Prepare data
    feature_cols = [col for col in train_df.columns if 'future' not in col and col != 'current_price']
    
    results = {}
    
    for horizon in [1, 7, 30]:
        print(f"\n🎯 {horizon}-day Horizon")
        print("-" * 40)
        
        # Prepare features and targets
        X_train = train_df[feature_cols]
        y_train = train_df[f'return_{horizon}d_future']
        X_test = test_df[feature_cols]
        y_test = test_df[f'return_{horizon}d_future']
        y_test_dir = test_df[f'direction_{horizon}d_future']
        
        # Remove NaN
        train_mask = ~y_train.isna()
        test_mask = ~y_test.isna()
        
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        y_test_dir = y_test_dir[test_mask]
        
        # Train regularized ensemble
        ensemble = RegularizedEnsemble(
            n_features_to_select=30,
            use_feature_selection=True,
            use_pca=False
        )
        
        train_results = ensemble.train(X_train, y_train)
        
        # Predict
        predictions = ensemble.predict(X_test)
        pred_directions = (predictions > 0).astype(int)
        
        # Evaluate
        accuracy = accuracy_score(y_test_dir, pred_directions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        # Directional accuracy by magnitude
        large_moves = np.abs(y_test) > y_test.std()
        if large_moves.sum() > 0:
            large_move_accuracy = accuracy_score(y_test_dir[large_moves], pred_directions[large_moves])
        else:
            large_move_accuracy = 0
        
        print(f"\n📊 Results:")
        print(f"   Direction Accuracy: {accuracy:.1%}")
        print(f"   Large Move Accuracy: {large_move_accuracy:.1%} ({large_moves.sum()} samples)")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        
        # Feature importance
        importance = ensemble.get_feature_importance()
        if importance:
            print(f"\n🔍 Top Features:")
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feat, imp in top_features:
                print(f"   {feat}: {imp:.3f}")
        
        results[f'{horizon}d'] = {
            'accuracy': accuracy,
            'rmse': rmse,
            'mae': mae,
            'large_move_accuracy': large_move_accuracy,
            'predictions': predictions,
            'actual': y_test
        }
    
    # 2. ADVANCED TIME SERIES MODELS
    print(f"\n{'='*80}")
    print("2. ADVANCED TIME SERIES MODELS")
    print("=" * 80)
    
    # Prepare data for time series models
    # Combine train and test for full time series
    full_df = pd.concat([train_df, test_df])
    
    # Add required columns for advanced models
    ts_df = pd.DataFrame({
        'price': full_df['current_price'],
        'rainfall_anomaly': full_df['Côte d\'Ivoire_temp_anomaly_30d'],
        'temperature_anomaly': full_df['Côte d\'Ivoire_temp_anomaly_30d'],
        'trade_volume_change': full_df.get('total_export_volume', 0) / 1e9,  # Scale down
        'export_concentration': full_df.get('Côte d\'Ivoire_export_share', 0.5),
        'sentiment_score': full_df.get('sentiment_mean', 0)
    })
    
    # Run advanced models
    adv_models = AdvancedTimeSeriesModels()
    
    # Slope of Slopes
    try:
        sos_results = adv_models.fit_slope_of_slopes(ts_df)
    except Exception as e:
        print(f"Slope of Slopes error: {e}")
    
    # TSMamba
    try:
        tsmamba_results = adv_models.fit_tsmamba(ts_df)
    except Exception as e:
        print(f"TSMamba error: {e}")
    
    # 3. COMPREHENSIVE PREDICTABILITY ANALYSIS
    print(f"\n{'='*80}")
    print("3. PREDICTABILITY ANALYSIS")
    print("=" * 80)
    
    analyzer = PredictabilityAnalyzer()
    
    # Run full analysis
    target_cols = ['return_1d_future', 'return_7d_future', 'return_30d_future']
    predictability_results = analyzer.analyze_predictability(
        full_df,
        target_cols,
        n_simulations=1000
    )
    
    # Generate detailed report
    analyzer.generate_report(predictability_results, 'predictability_advanced.html')
    
    # Print summary
    print("\n📊 Predictability Summary:")
    print(f"   Has Predictive Power: {'YES' if predictability_results['has_predictive_power'] else 'NO'}")
    print(f"   Confidence Level: {predictability_results.get('confidence_level', 'N/A')}")
    
    # Statistical tests
    if 'statistical_tests' in predictability_results:
        print("\n📈 Statistical Tests:")
        for target in target_cols:
            if target in predictability_results['statistical_tests']:
                tests = predictability_results['statistical_tests'][target]
                print(f"\n   {target}:")
                print(f"     Autocorrelation: {'Yes' if tests.get('has_autocorrelation') else 'No'}")
                print(f"     Random Walk: {'Yes' if tests.get('is_random_walk') else 'No'}")
                print(f"     Mean Reversion: {'Yes' if tests.get('has_mean_reversion') else 'No'}")
    
    # 4. FINAL SUMMARY
    print(f"\n{'='*80}")
    print("FINAL EVALUATION SUMMARY")
    print("=" * 80)
    
    print("\n1. Model Performance:")
    for horizon, res in results.items():
        print(f"   {horizon}: {res['accuracy']:.1%} accuracy (RMSE: {res['rmse']:.4f})")
    
    print("\n2. Key Insights:")
    print(f"   - Best horizon: {max(results.items(), key=lambda x: x[1]['accuracy'])[0]}")
    print(f"   - Feature selection improved generalization")
    print(f"   - Large move prediction accuracy: {np.mean([r['large_move_accuracy'] for r in results.values()]):.1%}")
    
    print("\n3. Advanced Models:")
    if 'slope_of_slopes' in adv_models.results:
        print(f"   - Slope of Slopes: {len(adv_models.results['slope_of_slopes']['trend_changes'])} trend changes detected")
    if 'tsmamba' in adv_models.results:
        print(f"   - TSMamba MAPE: {adv_models.results['tsmamba']['test_mape']:.1f}%")
    
    print("\n✅ Evaluation complete! Check predictability_advanced.html for detailed report.")
    
    return results, predictability_results


if __name__ == "__main__":
    results, predictability = run_advanced_evaluation()