#!/usr/bin/env python3
"""
Run Full Historical Backtesting with Regularized Models
Shows predicted vs actual for ENTIRE time period with accuracy metrics
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from src.models.regularized_ensemble import RegularizedEnsemble
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def run_full_backtesting():
    """Run complete historical backtesting"""
    
    print("🚀 FULL HISTORICAL BACKTESTING")
    print("=" * 80)
    
    # Load comprehensive dataset
    train_df = pd.read_csv('data/processed/comprehensive_train.csv', index_col='date', parse_dates=True)
    test_df = pd.read_csv('data/processed/comprehensive_test.csv', index_col='date', parse_dates=True)
    
    # Combine for full history
    full_df = pd.concat([train_df, test_df]).sort_index()
    
    print(f"📊 Full Dataset: {len(full_df)} samples")
    print(f"   Period: {full_df.index[0].date()} to {full_df.index[-1].date()}")
    print(f"   Duration: {(full_df.index[-1] - full_df.index[0]).days} days")
    
    # Prepare features
    feature_cols = [col for col in full_df.columns if 'future' not in col and col != 'current_price']
    
    # Results storage
    all_results = {
        'date': [],
        'actual_price': [],
        'actual_1d_return': [],
        'actual_7d_return': [], 
        'actual_30d_return': [],
        'pred_1d_return': [],
        'pred_7d_return': [],
        'pred_30d_return': [],
        'pred_1d_dir': [],
        'pred_7d_dir': [],
        'pred_30d_dir': [],
        'actual_1d_dir': [],
        'actual_7d_dir': [],
        'actual_30d_dir': []
    }
    
    # WALK-FORWARD BACKTESTING
    print("\n🔄 Running Walk-Forward Backtesting...")
    print("   (Training on all data up to each point, predicting next period)")
    
    min_train_size = 100  # Need at least 100 samples to train
    horizons = [1, 7, 30]
    
    # Initialize models
    models = {
        h: RegularizedEnsemble(
            n_features_to_select=30,
            use_feature_selection=True,
            use_pca=False
        ) for h in horizons
    }
    
    # Walk-forward analysis
    for i in range(min_train_size, len(full_df)):
        if i % 20 == 0:  # Progress update
            print(f"   Processing: {i}/{len(full_df)} ({i/len(full_df)*100:.1f}%)")
        
        # Current date
        current_date = full_df.index[i]
        
        # Get all data up to this point
        train_data = full_df.iloc[:i]
        test_point = full_df.iloc[i:i+1]
        
        # Store date and price
        all_results['date'].append(current_date)
        all_results['actual_price'].append(test_point['current_price'].values[0])
        
        # Predict for each horizon
        for h in horizons:
            try:
                # Prepare training data
                X_train = train_data[feature_cols]
                y_train = train_data[f'return_{h}d_future']
                
                # Remove NaN
                train_mask = ~y_train.isna()
                X_train = X_train[train_mask]
                y_train = y_train[train_mask]
                
                if len(X_train) < 50:  # Skip if not enough data
                    all_results[f'pred_{h}d_return'].append(np.nan)
                    all_results[f'pred_{h}d_dir'].append(np.nan)
                    continue
                
                # Train model (retrain every 10 days to save time)
                if i % 10 == 0 or i == min_train_size:
                    models[h].train(X_train, y_train)
                
                # Predict
                X_test = test_point[feature_cols]
                pred = models[h].predict(X_test)[0]
                pred_dir = 1 if pred > 0 else 0
                
                all_results[f'pred_{h}d_return'].append(pred)
                all_results[f'pred_{h}d_dir'].append(pred_dir)
                
            except Exception as e:
                all_results[f'pred_{h}d_return'].append(np.nan)
                all_results[f'pred_{h}d_dir'].append(np.nan)
        
        # Store actual values
        for h in horizons:
            actual_return = test_point[f'return_{h}d_future'].values[0] if f'return_{h}d_future' in test_point else np.nan
            actual_dir = test_point[f'direction_{h}d_future'].values[0] if f'direction_{h}d_future' in test_point else np.nan
            
            all_results[f'actual_{h}d_return'].append(actual_return)
            all_results[f'actual_{h}d_dir'].append(actual_dir)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df.set_index('date', inplace=True)
    
    # Calculate accuracy metrics
    print("\n" + "="*80)
    print("📊 BACKTESTING RESULTS - ACCURACY BY HORIZON")
    print("="*80)
    
    for h in horizons:
        # Remove NaN values for accuracy calculation
        mask = ~(results_df[f'pred_{h}d_dir'].isna() | results_df[f'actual_{h}d_dir'].isna())
        
        if mask.sum() > 0:
            accuracy = (results_df.loc[mask, f'pred_{h}d_dir'] == results_df.loc[mask, f'actual_{h}d_dir']).mean()
            n_samples = mask.sum()
            
            # Calculate by year
            results_df['year'] = results_df.index.year
            yearly_acc = results_df[mask].groupby('year').apply(
                lambda x: (x[f'pred_{h}d_dir'] == x[f'actual_{h}d_dir']).mean()
            )
            
            print(f"\n{h}-DAY HORIZON:")
            print(f"  Overall Accuracy: {accuracy:.1%} ({n_samples} predictions)")
            print(f"  Accuracy by Year:")
            for year, acc in yearly_acc.items():
                n_year = (results_df[mask]['year'] == year).sum()
                print(f"    {year}: {acc:.1%} ({n_year} predictions)")
    
    # Save results
    results_df.to_csv('data/processed/backtesting_results_full.csv')
    print(f"\n💾 Backtesting results saved to: data/processed/backtesting_results_full.csv")
    
    # Create visualization
    print("\n📈 Creating visualization...")
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), facecolor='#272b30')
    
    for i, h in enumerate(horizons):
        ax = axes[i]
        ax.set_facecolor('#272b30')
        
        # Plot actual vs predicted returns
        mask = ~(results_df[f'actual_{h}d_return'].isna() | results_df[f'pred_{h}d_return'].isna())
        
        if mask.sum() > 0:
            ax.scatter(results_df.loc[mask, f'actual_{h}d_return'], 
                      results_df.loc[mask, f'pred_{h}d_return'],
                      alpha=0.5, color='#6f42c1', s=10)
            
            # Add diagonal line
            min_val = min(results_df.loc[mask, f'actual_{h}d_return'].min(), 
                         results_df.loc[mask, f'pred_{h}d_return'].min())
            max_val = max(results_df.loc[mask, f'actual_{h}d_return'].max(),
                         results_df.loc[mask, f'pred_{h}d_return'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'w--', alpha=0.5)
            
            # Calculate correlation
            corr = results_df.loc[mask, [f'actual_{h}d_return', f'pred_{h}d_return']].corr().iloc[0, 1]
            
            ax.set_xlabel(f'Actual {h}d Return', color='white', fontsize=12)
            ax.set_ylabel(f'Predicted {h}d Return', color='white', fontsize=12)
            ax.set_title(f'{h}-Day Horizon (Correlation: {corr:.3f})', color='white', fontsize=14)
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.grid(True, alpha=0.2, color='white')
    
    plt.tight_layout()
    plt.savefig('backtesting_results_visual.png', dpi=150, facecolor='#272b30')
    print("📊 Visualization saved to: backtesting_results_visual.png")
    
    # Summary statistics
    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    
    for h in horizons:
        mask = ~(results_df[f'pred_{h}d_return'].isna() | results_df[f'actual_{h}d_return'].isna())
        
        if mask.sum() > 0:
            # Calculate RMSE
            rmse = np.sqrt(((results_df.loc[mask, f'pred_{h}d_return'] - 
                           results_df.loc[mask, f'actual_{h}d_return'])**2).mean())
            
            # Calculate directional accuracy
            dir_mask = ~(results_df[f'pred_{h}d_dir'].isna() | results_df[f'actual_{h}d_dir'].isna())
            dir_accuracy = (results_df.loc[dir_mask, f'pred_{h}d_dir'] == 
                          results_df.loc[dir_mask, f'actual_{h}d_dir']).mean()
            
            print(f"\n{h}-Day Horizon:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Direction Accuracy: {dir_accuracy:.1%}")
            print(f"  Predictions Made: {mask.sum()}")
    
    return results_df


if __name__ == "__main__":
    results = run_full_backtesting()
    print("\n✅ Full historical backtesting complete!")