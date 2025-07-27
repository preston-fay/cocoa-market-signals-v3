#!/usr/bin/env python3
"""
Quick test of model performance with comprehensive dataset
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.models.multi_source_predictor import MultiSourcePredictor
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def quick_test():
    """Quick test of model performance"""
    
    print("🚀 Quick Model Test with Comprehensive Dataset")
    print("=" * 60)
    
    # Load data
    train_df = pd.read_csv('data/processed/comprehensive_train.csv', index_col='date', parse_dates=True)
    test_df = pd.read_csv('data/processed/comprehensive_test.csv', index_col='date', parse_dates=True)
    
    print(f"📊 Dataset: {len(train_df)} train, {len(test_df)} test samples")
    
    # Prepare features
    feature_cols = [col for col in train_df.columns if 'future' not in col and col != 'current_price']
    
    # Test 7-day prediction (usually best performance)
    X_train = train_df[feature_cols]
    y_train = train_df['return_7d_future']
    X_test = test_df[feature_cols]
    y_test = test_df['return_7d_future']
    y_test_dir = test_df['direction_7d_future']
    
    # Remove NaN
    train_mask = ~y_train.isna()
    test_mask = ~y_test.isna()
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    y_test_dir = y_test_dir[test_mask]
    
    print(f"\n🎯 Testing 7-day horizon prediction")
    print(f"   Features: {len(feature_cols)}")
    
    # Train model
    predictor = MultiSourcePredictor()
    train_results = predictor.train(X_train, y_train)
    
    print(f"\n📊 Training Results:")
    for model, metrics in train_results.items():
        if isinstance(metrics, dict) and 'r2' in metrics:
            print(f"   {model}: R² = {metrics['r2']:.3f}")
    
    # Predict
    predictions = predictor.predict(X_test)
    pred_directions = (predictions > 0).astype(int)
    
    # Evaluate
    accuracy = accuracy_score(y_test_dir, pred_directions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    # Calculate percentage of positive predictions
    pct_positive_pred = pred_directions.mean()
    pct_positive_actual = y_test_dir.mean()
    
    print(f"\n✅ Test Results (7-day horizon):")
    print(f"   Direction Accuracy: {accuracy:.1%}")
    print(f"   Return RMSE: {rmse:.4f}")
    print(f"   % Positive Predicted: {pct_positive_pred:.1%}")
    print(f"   % Positive Actual: {pct_positive_actual:.1%}")
    
    # Compare with baseline
    baseline_accuracy = max(pct_positive_actual, 1 - pct_positive_actual)
    improvement = accuracy - baseline_accuracy
    
    print(f"\n📊 vs Baseline (always predict majority class):")
    print(f"   Baseline: {baseline_accuracy:.1%}")
    print(f"   Improvement: {improvement:+.1%}")
    
    # Show some predictions
    print(f"\n🔍 Sample Predictions (first 10):")
    print("Date         | Actual Dir | Pred Dir | Actual Return | Pred Return")
    print("-" * 65)
    for i in range(min(10, len(y_test))):
        date = X_test.index[i].strftime('%Y-%m-%d')
        print(f"{date} |     {'Up' if y_test_dir.iloc[i] else 'Down'}     |   {'Up' if pred_directions[i] else 'Down'}    | {y_test.iloc[i]:+.4f}      | {predictions[i]:+.4f}")
    
    return accuracy, rmse


if __name__ == "__main__":
    accuracy, rmse = quick_test()
    print(f"\n✅ Quick test complete!")
    print(f"🎯 Final: {accuracy:.1%} accuracy with {rmse:.4f} RMSE")