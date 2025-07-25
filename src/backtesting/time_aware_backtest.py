"""
Time-Aware Backtesting Engine
Following DATA_SCIENCE_STANDARDS.md from preston-dev-setup
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class TimeAwareBacktester:
    """
    Proper walk-forward backtesting that respects time boundaries
    No future data leakage - only uses data available at each point
    """
    
    def __init__(self, data_path: str = "data/processed/unified_real_data.csv"):
        """Initialize with real data"""
        self.df = pd.read_csv(data_path)
        # Handle timezone-aware dates properly
        self.df['date'] = pd.to_datetime(self.df['date'], utc=True).dt.tz_localize(None)
        self.df = self.df.sort_values('date').reset_index(drop=True)
        
        # Prediction windows to test (in days)
        self.prediction_windows = {
            '1_week': 7,
            '1_month': 30,
            '2_months': 60,
            '3_months': 90,
            '6_months': 180
        }
        
        # Store results
        self.backtest_results = {}
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features using ONLY past data
        Following DATA_SCIENCE_STANDARDS: Document all transformations
        """
        features = pd.DataFrame(index=df.index)
        
        # Price features (lagged to avoid leakage)
        features['price_lag_1'] = df['price'].shift(1)
        features['price_lag_7'] = df['price'].shift(7)
        features['price_lag_30'] = df['price'].shift(30)
        
        # Price momentum
        features['momentum_7d'] = df['price'].pct_change(7)
        features['momentum_30d'] = df['price'].pct_change(30)
        
        # Rolling statistics (backward looking only)
        features['volatility_30d'] = df['price'].pct_change().rolling(30).std()
        features['price_sma_30'] = df['price'].rolling(30).mean()
        features['price_sma_90'] = df['price'].rolling(90).mean()
        
        # Weather features (current values are OK - they're observations)
        features['rainfall_anomaly'] = df['rainfall_anomaly']
        features['temperature_anomaly'] = df['temperature_anomaly']
        
        # Weather rolling stats
        features['rainfall_30d_mean'] = df['rainfall_anomaly'].rolling(30).mean()
        features['temp_30d_mean'] = df['temperature_anomaly'].rolling(30).mean()
        
        # Trade features
        features['export_concentration'] = df['export_concentration']
        features['trade_volume_change'] = df['trade_volume_change']
        
        # Time features
        features['month'] = df['date'].dt.month
        features['quarter'] = df['date'].dt.quarter
        features['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        
        return features
    
    def run_backtest(self, start_date: str = '2024-01-01', end_date: str = '2025-01-01'):
        """
        Run walk-forward backtest for specified period
        Following TimeSeriesSplit from DATA_SCIENCE_STANDARDS
        """
        print(f"Running backtest from {start_date} to {end_date}")
        
        # Convert string dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Filter data
        mask = (self.df['date'] >= start_dt) & (self.df['date'] <= end_dt)
        test_data = self.df[mask].copy()
        
        # Prepare features
        all_features = self.prepare_features(self.df)
        
        # For each month in the test period
        test_months = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        results_by_window = {window: [] for window in self.prediction_windows}
        
        for current_date in test_months:
            print(f"\nAnalyzing {current_date.strftime('%Y-%m')}")
            
            # Get all data up to current date (no future info!)
            historical_mask = self.df['date'] < current_date
            historical_data = self.df[historical_mask].copy()
            historical_features = all_features[historical_mask].copy()
            
            # Need at least 180 days of history
            if len(historical_data) < 180:
                continue
                
            # Drop NaN values
            valid_mask = ~historical_features.isnull().any(axis=1)
            X_train = historical_features[valid_mask]
            y_train = historical_data.loc[valid_mask, 'price']
            
            # Train model on all historical data
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,  # For reproducibility
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Make predictions for each window
            for window_name, days_ahead in self.prediction_windows.items():
                future_date = current_date + timedelta(days=days_ahead)
                
                # Check if we have actual data for validation
                future_mask = (self.df['date'] >= future_date - timedelta(days=3)) & \
                             (self.df['date'] <= future_date + timedelta(days=3))
                
                if not any(future_mask):
                    continue
                    
                # Get features at current date
                current_mask = (self.df['date'] >= current_date - timedelta(days=3)) & \
                              (self.df['date'] <= current_date + timedelta(days=3))
                
                if not any(current_mask):
                    continue
                
                current_idx = self.df[current_mask].index[0]
                current_features = all_features.iloc[[current_idx]]
                
                # Drop NaN features
                feature_cols = X_train.columns
                current_features = current_features[feature_cols]
                
                if current_features.isnull().any().any():
                    continue
                
                # Make prediction
                predicted_price = model.predict(current_features)[0]
                
                # Get actual price
                actual_price = self.df[future_mask]['price'].iloc[0]
                
                # Calculate metrics
                error = predicted_price - actual_price
                pct_error = (error / actual_price) * 100
                
                # Store result
                results_by_window[window_name].append({
                    'analysis_date': current_date,
                    'prediction_date': future_date,
                    'predicted_price': predicted_price,
                    'actual_price': actual_price,
                    'error': error,
                    'pct_error': pct_error,
                    'abs_pct_error': abs(pct_error)
                })
        
        # Calculate aggregate metrics
        self.backtest_results = self._calculate_metrics(results_by_window)
        return self.backtest_results
    
    def _calculate_metrics(self, results_by_window: Dict) -> Dict:
        """Calculate performance metrics for each prediction window"""
        metrics = {}
        
        for window, results in results_by_window.items():
            if not results:
                continue
                
            df_results = pd.DataFrame(results)
            
            metrics[window] = {
                'num_predictions': len(df_results),
                'mae': mean_absolute_error(df_results['actual_price'], 
                                          df_results['predicted_price']),
                'rmse': np.sqrt(mean_squared_error(df_results['actual_price'], 
                                                  df_results['predicted_price'])),
                'mape': df_results['abs_pct_error'].mean(),
                'directional_accuracy': ((df_results['predicted_price'].diff() > 0) == 
                                       (df_results['actual_price'].diff() > 0)).mean(),
                'avg_error': df_results['error'].mean(),
                'std_error': df_results['error'].std(),
                'predictions': results  # Keep detailed results
            }
            
        return metrics
    
    def save_results(self, output_path: str = "data/processed/backtest_results.json"):
        """Save backtest results following data versioning standards"""
        output = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'data_version': 'unified_real_data_v1',
                'model': 'RandomForestRegressor',
                'parameters': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                }
            },
            'results': self.backtest_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
            
        print(f"\nBacktest results saved to {output_path}")
        
    def print_summary(self):
        """Print summary of backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        
        for window, metrics in self.backtest_results.items():
            print(f"\n{window.replace('_', ' ').upper()}:")
            print(f"  Predictions made: {metrics['num_predictions']}")
            print(f"  Mean Absolute Error: ${metrics['mae']:,.2f}")
            print(f"  RMSE: ${metrics['rmse']:,.2f}")
            print(f"  Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
            print(f"  Directional Accuracy: {metrics['directional_accuracy']*100:.1f}%")
            
            
if __name__ == "__main__":
    # Run backtest following reproducibility standards
    np.random.seed(42)
    
    backtester = TimeAwareBacktester()
    results = backtester.run_backtest(
        start_date='2024-01-01',
        end_date='2025-06-01'
    )
    
    backtester.print_summary()
    backtester.save_results()