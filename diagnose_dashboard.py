#!/usr/bin/env python3
"""
Diagnose dashboard issues and provide detailed report
"""
import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
import json
from datetime import datetime

def diagnose_dashboard():
    """Diagnose what's wrong with the dashboard"""
    print("🔍 Dashboard Diagnostic Report")
    print("=" * 60)
    
    issues = []
    
    # 1. Check data files
    print("\n1. Checking Data Files:")
    
    # Backtesting results
    try:
        results_df = pd.read_csv('data/processed/backtesting_results_full.csv')
        print(f"   ✅ Backtesting results: {len(results_df)} rows")
        
        # Check if it has meaningful data
        if 'actual_7d_return' in results_df.columns:
            non_null = results_df['actual_7d_return'].notna().sum()
            print(f"      - Non-null predictions: {non_null}")
            if non_null < 10:
                issues.append("Very few actual predictions in backtesting results")
        else:
            issues.append("Missing actual_7d_return column in backtesting results")
            
    except Exception as e:
        print(f"   ❌ Backtesting results: {e}")
        issues.append("Cannot load backtesting results")
    
    # Training data
    try:
        train_df = pd.read_csv('data/processed/comprehensive_train.csv')
        print(f"   ✅ Training data: {len(train_df)} rows")
    except Exception as e:
        print(f"   ❌ Training data: {e}")
        issues.append("Cannot load training data")
    
    # News data
    try:
        with open('data/historical/news/real_cocoa_news.json', 'r') as f:
            news_data = json.load(f)
        print(f"   ✅ News data: {len(news_data)} articles")
    except Exception as e:
        print(f"   ❌ News data: {e}")
        issues.append("Cannot load news data")
    
    # 2. Check signal generation logic
    print("\n2. Checking Signal Generation:")
    
    try:
        # Load the showcase module
        from src.dashboard.app_showcase import MarketSignalShowcase
        showcase = MarketSignalShowcase()
        
        print(f"   - Signals found: {len(showcase.signals)}")
        
        if len(showcase.signals) == 0:
            issues.append("No signals generated - check signal detection logic")
            
            # Debug signal generation
            print("\n   Debugging signal generation:")
            if hasattr(showcase, 'results_df'):
                df = showcase.results_df
                
                # Check for large moves
                if 'actual_7d_return' in df.columns:
                    large_moves = df[df['actual_7d_return'].abs() > 0.05]
                    print(f"      - Large moves (>5%): {len(large_moves)}")
                    
                    if len(large_moves) > 0:
                        # Check predictions
                        if 'pred_7d_return' in df.columns:
                            correct_predictions = large_moves[
                                (large_moves['actual_7d_return'] * large_moves['pred_7d_return']) > 0
                            ]
                            print(f"      - Correctly predicted: {len(correct_predictions)}")
                        else:
                            issues.append("Missing pred_7d_return column")
                else:
                    issues.append("Missing actual_7d_return column")
                    
        else:
            print(f"   ✅ Signals generated successfully")
            # Show sample signal
            if showcase.signals:
                signal = showcase.signals[0]
                print(f"      Sample signal: {signal['date']} - {signal['type']} - {signal['magnitude']:.1%}")
                
    except Exception as e:
        print(f"   ❌ Error loading showcase: {e}")
        issues.append(f"Cannot load showcase module: {str(e)}")
    
    # 3. Explain Accuracy
    print("\n3. About 'Accuracy %':")
    print("   The accuracy percentage represents:")
    print("   - DIRECTIONAL ACCURACY: Did we predict UP or DOWN correctly?")
    print("   - For 7-day predictions: If price goes up/down in 7 days, did we predict that?")
    print("   - 62.2% means we're right about direction 62.2% of the time")
    print("   - 92.3% on 'large moves' means when price moves >5%, we're right 92.3% of the time")
    print("   - This is valuable because large moves are where money is made/lost")
    
    # 4. Summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if issues:
        print("\n❌ Issues Found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("\n✅ No major issues found")
    
    print("\n💡 Recommendations:")
    print("   1. Ensure backtesting_results_full.csv has predicted values")
    print("   2. Check that signal detection thresholds are appropriate")
    print("   3. Verify news data has dates that match the analysis period")
    
    return issues


if __name__ == "__main__":
    issues = diagnose_dashboard()
    
    if issues:
        print("\n🔧 To fix the dashboard:")
        print("   1. Re-run backtesting to generate predictions")
        print("   2. Ensure all data files are properly formatted")
        print("   3. Check signal generation thresholds")