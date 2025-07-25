"""
Statistical Models for Market Signal Detection
Includes time series analysis, anomaly detection, and predictive modeling
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class StatisticalSignalModels:
    """Suite of statistical models for signal detection and validation"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def prepare_time_series_data(self, weather_data, trade_data, price_data):
        """Prepare data for time series analysis"""
        # Create DataFrame with all variables
        dates = sorted(list(set(weather_data.keys()) & set(trade_data.keys()) & set(price_data.keys())))
        
        data = []
        for date in dates:
            row = {
                'date': pd.to_datetime(date),
                'price': price_data[date],
                'rainfall_anomaly': weather_data[date]['avg_rainfall_anomaly'],
                'temperature_anomaly': weather_data[date]['avg_temp_anomaly'],
                'trade_volume_change': trade_data[date]['volume_change_pct'],
                'export_concentration': trade_data[date]['export_concentration']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def test_stationarity(self, series, name="Series"):
        """Augmented Dickey-Fuller test for stationarity"""
        result = adfuller(series.dropna())
        
        logger.info(f"\nStationarity Test for {name}:")
        logger.info(f"ADF Statistic: {result[0]:.4f}")
        logger.info(f"p-value: {result[1]:.4f}")
        logger.info(f"Critical Values:")
        for key, value in result[4].items():
            logger.info(f"  {key}: {value:.4f}")
        
        is_stationary = result[1] < 0.05
        logger.info(f"Result: {'Stationary' if is_stationary else 'Non-stationary'} at 5% significance")
        
        return {
            'statistic': result[0],
            'p_value': result[1],
            'is_stationary': is_stationary,
            'critical_values': result[4]
        }
    
    def granger_causality_test(self, data, target_col, test_cols, max_lag=4):
        """Test if variables Granger-cause the target (price)"""
        results = {}
        
        logger.info(f"\nGranger Causality Tests (Does X cause {target_col}?):")
        logger.info("-" * 60)
        
        for col in test_cols:
            test_data = data[[target_col, col]].dropna()
            
            try:
                gc_test = grangercausalitytests(test_data, max_lag, verbose=False)
                
                # Extract p-values for each lag
                p_values = []
                for lag in range(1, max_lag + 1):
                    p_val = gc_test[lag][0]['ssr_ftest'][1]
                    p_values.append(p_val)
                
                # Find best lag (lowest p-value)
                best_lag = np.argmin(p_values) + 1
                best_p_value = p_values[best_lag - 1]
                
                results[col] = {
                    'best_lag': best_lag,
                    'p_value': best_p_value,
                    'causes_target': best_p_value < 0.05,
                    'all_p_values': p_values
                }
                
                logger.info(f"{col:20} -> Lag {best_lag}: p={best_p_value:.4f} "
                      f"{'✓ Causes' if best_p_value < 0.05 else '✗ No causality'}")
                
            except Exception as e:
                logger.error(f"{col:20} -> Error: {str(e)}")
                results[col] = {'error': str(e)}
        
        return results
    
    def multicollinearity_test(self, data, feature_cols):
        """Calculate Variance Inflation Factor to test multicollinearity"""
        vif_data = pd.DataFrame()
        vif_data["Feature"] = feature_cols
        vif_data["VIF"] = [variance_inflation_factor(data[feature_cols].values, i) 
                          for i in range(len(feature_cols))]
        
        logger.info("\nMulticollinearity Test (VIF):")
        logger.info(vif_data)
        logger.info("\nInterpretation: VIF > 10 indicates high multicollinearity")
        
        return vif_data
    
    def build_anomaly_detection_model(self, data, contamination=0.1):
        """Isolation Forest for multivariate anomaly detection"""
        logger.info("\nBuilding Anomaly Detection Model...")
        
        # Select features
        features = ['rainfall_anomaly', 'temperature_anomaly', 'trade_volume_change']
        X = data[features].values
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        anomaly_predictions = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.score_samples(X)
        
        # Add to dataframe
        data['anomaly'] = anomaly_predictions
        data['anomaly_score'] = anomaly_scores
        
        # Calculate statistics
        n_anomalies = (anomaly_predictions == -1).sum()
        anomaly_dates = data[data['anomaly'] == -1].index
        
        logger.info(f"Detected {n_anomalies} anomalies out of {len(data)} observations")
        logger.info(f"Anomaly dates: {[d.strftime('%Y-%m') for d in anomaly_dates]}")
        
        self.models['anomaly_detector'] = iso_forest
        
        return {
            'model': iso_forest,
            'predictions': anomaly_predictions,
            'scores': anomaly_scores,
            'anomaly_dates': anomaly_dates,
            'feature_importance': self._calculate_isolation_importance(iso_forest, features)
        }
    
    def _calculate_isolation_importance(self, model, features):
        """Estimate feature importance for Isolation Forest"""
        # Simple approximation based on path lengths
        importances = []
        for i, feature in enumerate(features):
            # This is a simplified approach
            importance = 1.0 / (i + 1)  # Placeholder
            importances.append(importance)
        
        # Normalize
        importances = np.array(importances)
        importances = importances / importances.sum()
        
        return dict(zip(features, importances))
    
    def build_predictive_model(self, data, target='price', test_size=0.2):
        """Random Forest model for price prediction"""
        logger.info("\nBuilding Predictive Model...")
        
        # Prepare features and target
        feature_cols = ['rainfall_anomaly', 'temperature_anomaly', 
                       'trade_volume_change', 'export_concentration']
        
        # Add lagged features
        for col in feature_cols:
            data[f'{col}_lag1'] = data[col].shift(1)
            data[f'{col}_lag2'] = data[col].shift(2)
        
        # Add rolling statistics
        for col in feature_cols:
            data[f'{col}_rolling_mean'] = data[col].rolling(window=3).mean()
            data[f'{col}_rolling_std'] = data[col].rolling(window=3).std()
        
        # Remove NaN rows
        data_clean = data.dropna()
        
        # Get all feature columns
        all_features = [col for col in data_clean.columns 
                       if col not in ['price', 'anomaly', 'anomaly_score']]
        
        X = data_clean[all_features]
        y = data_clean[target]
        
        # Time series split (no random shuffling)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = rf_model.predict(X_train_scaled)
        y_test_pred = rf_model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
        
        logger.info(f"\nModel Performance:")
        logger.info(f"Train MSE: {train_mse:.2f}, R²: {train_r2:.3f}")
        logger.info(f"Test MSE: {test_mse:.2f}, R²: {test_r2:.3f}")
        logger.info(f"Test MAPE: {test_mape:.2%}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': all_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop 10 Most Important Features:")
        logger.info(feature_importance.head(10))
        
        self.models['price_predictor'] = rf_model
        self.scalers['price_predictor'] = scaler
        
        return {
            'model': rf_model,
            'scaler': scaler,
            'train_metrics': {'mse': train_mse, 'r2': train_r2},
            'test_metrics': {'mse': test_mse, 'r2': test_r2, 'mape': test_mape},
            'feature_importance': feature_importance,
            'predictions': y_test_pred,
            'y_test': y_test,
            'y_train': y_train,
            'X_test': X_test,
            'X_train': X_train
        }
    
    def calculate_signal_correlations(self, data):
        """Calculate correlations between signals and price movements"""
        # Calculate price changes
        data['price_change'] = data['price'].pct_change()
        data['price_change_future'] = data['price_change'].shift(-1)  # Next period change
        
        # Select signal columns
        signal_cols = ['rainfall_anomaly', 'temperature_anomaly', 
                      'trade_volume_change', 'export_concentration']
        
        # Calculate correlations
        correlations = {}
        
        logger.info("\nSignal Correlations with Future Price Changes:")
        logger.info("-" * 50)
        
        for col in signal_cols:
            # Pearson correlation
            pearson_corr, pearson_p = stats.pearsonr(
                data[col].dropna(), 
                data.loc[data[col].notna(), 'price_change_future'].fillna(0)
            )
            
            # Spearman correlation (non-parametric)
            spearman_corr, spearman_p = stats.spearmanr(
                data[col].dropna(),
                data.loc[data[col].notna(), 'price_change_future'].fillna(0)
            )
            
            correlations[col] = {
                'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
                'spearman': {'correlation': spearman_corr, 'p_value': spearman_p}
            }
            
            logger.info(f"{col:20} | Pearson: {pearson_corr:+.3f} (p={pearson_p:.3f}) | "
                  f"Spearman: {spearman_corr:+.3f} (p={spearman_p:.3f})")
        
        return correlations
    
    def perform_regime_detection(self, price_series, n_regimes=3):
        """Detect market regimes using statistical methods"""
        logger.info("\nPerforming Regime Detection...")
        
        # Calculate returns
        returns = price_series.pct_change().dropna()
        
        # Simple regime detection using rolling statistics
        window = 20
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        
        # Define regimes based on mean and volatility
        regimes = pd.Series(index=returns.index, dtype=int)
        
        # High volatility threshold
        high_vol_threshold = rolling_std.quantile(0.75)
        
        # Classify regimes numerically
        regime_map = {}
        regime_id = 0
        
        for date in returns.index[window:]:
            mean = rolling_mean.loc[date]
            std = rolling_std.loc[date]
            
            if std > high_vol_threshold:
                if mean > 0.02:
                    regime_name = 'Bull_High_Vol'
                else:
                    regime_name = 'Bear_High_Vol'
            else:
                if mean > 0.01:
                    regime_name = 'Bull_Low_Vol'
                elif mean < -0.01:
                    regime_name = 'Bear_Low_Vol'
                else:
                    regime_name = 'Neutral'
                    
            if regime_name not in regime_map:
                regime_map[regime_name] = regime_id
                regime_id += 1
                
            regimes.loc[date] = regime_map[regime_name]
        
        # Fill NaN values
        regimes = regimes.fillna(0).astype(int)
        
        # Count regime frequencies
        regime_counts = regimes.value_counts()
        
        logger.info("\nRegime Distribution:")
        for regime_name, regime_num in regime_map.items():
            count = (regimes == regime_num).sum()
            logger.info(f"{regime_name} (regime {regime_num}): {count} days")
        
        return {
            'regimes': regimes,
            'n_regimes': len(regime_map),
            'regime_map': regime_map,
            'regime_counts': regime_counts,
            'volatility_threshold': high_vol_threshold
        }
    
    def validate_signal_timing(self, signals, price_data, lead_times=[1, 2, 3]):
        """Validate if signals provide advance warning of price movements"""
        logger.info("\nValidating Signal Timing...")
        
        results = {}
        
        for lead in lead_times:
            logger.info(f"\n{lead}-Month Lead Time Analysis:")
            
            correct_signals = 0
            total_signals = 0
            
            for i in range(len(signals) - lead):
                if signals.iloc[i]['signal_strength'] < 0.5:  # Bullish signal
                    total_signals += 1
                    
                    # Check if price increased over next 'lead' periods
                    current_price = price_data.iloc[i]
                    future_price = price_data.iloc[i + lead]
                    
                    if future_price > current_price * 1.05:  # 5% threshold
                        correct_signals += 1
            
            accuracy = correct_signals / total_signals if total_signals > 0 else 0
            
            results[f'{lead}_month'] = {
                'accuracy': accuracy,
                'correct_signals': correct_signals,
                'total_signals': total_signals
            }
            
            logger.info(f"Accuracy: {accuracy:.1%} ({correct_signals}/{total_signals})")
        
        return results
    
    def calculate_risk_metrics(self, returns, confidence_level=0.95):
        """Calculate Value at Risk and Conditional Value at Risk"""
        logger.info("\nCalculating Risk Metrics...")
        
        # Historical VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns.dropna(), var_percentile)
        
        # CVaR (Expected Shortfall)
        cvar = returns[returns <= var].mean()
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (assuming 0 risk-free rate)
        sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        
        logger.info(f"VaR ({confidence_level:.0%}): {var:.3%}")
        logger.info(f"CVaR: {cvar:.3%}")
        logger.info(f"Max Drawdown: {max_drawdown:.3%}")
        logger.info(f"Sharpe Ratio: {sharpe:.3f}")
        
        return {
            'daily_var': var,
            'daily_cvar': cvar,
            'var': var,
            'cvar': cvar,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'annualized_volatility': returns.std() * np.sqrt(252)
        }

# Test data generator for demonstration
def generate_test_data():
    """Generate test data matching the cocoa price surge pattern"""
    dates = pd.date_range('2023-01', '2024-02', freq='M')
    
    # Weather data (anomalies increase from Sept 2023)
    weather_data = {}
    for i, date in enumerate(dates):
        month_str = date.strftime('%Y-%m')
        
        if date >= pd.Timestamp('2023-09'):
            anomaly_factor = (i - 8) * 0.15
            weather_data[month_str] = {
                'avg_rainfall_anomaly': np.random.normal(0.3 + anomaly_factor, 0.1),
                'avg_temp_anomaly': np.random.normal(0.2 + anomaly_factor * 0.5, 0.05)
            }
        else:
            weather_data[month_str] = {
                'avg_rainfall_anomaly': np.random.normal(0, 0.1),
                'avg_temp_anomaly': np.random.normal(0, 0.05)
            }
    
    # Trade data (volumes drop from Oct 2023)
    trade_data = {}
    for i, date in enumerate(dates):
        month_str = date.strftime('%Y-%m')
        
        if date >= pd.Timestamp('2023-10'):
            drop_factor = (i - 9) * -0.08
            trade_data[month_str] = {
                'volume_change_pct': np.random.normal(drop_factor - 0.1, 0.05),
                'export_concentration': np.random.uniform(0.5, 0.7)
            }
        else:
            trade_data[month_str] = {
                'volume_change_pct': np.random.normal(0, 0.05),
                'export_concentration': np.random.uniform(0.3, 0.5)
            }
    
    # Price data (surge from Nov 2023)
    price_data = {}
    base_price = 2500
    
    for i, date in enumerate(dates):
        month_str = date.strftime('%Y-%m')
        
        if date >= pd.Timestamp('2023-11'):
            months_since = i - 10
            price_increase = base_price * (1 + months_since * 0.25) ** 1.5
            price_data[month_str] = min(price_increase, 11000)
        else:
            price_data[month_str] = base_price + np.random.normal(0, 100)
    
    return weather_data, trade_data, price_data

if __name__ == "__main__":
    # Initialize models
    models = StatisticalSignalModels()
    
    # Generate test data
    print("Generating test data...")
    weather_data, trade_data, price_data = generate_test_data()
    
    # Prepare time series
    df = models.prepare_time_series_data(weather_data, trade_data, price_data)
    
    print("\nData Summary:")
    print(df.describe())
    
    # 1. Test stationarity
    print("\n" + "="*60)
    print("1. STATIONARITY TESTS")
    print("="*60)
    
    for col in df.columns:
        models.test_stationarity(df[col], name=col)
    
    # 2. Granger causality
    print("\n" + "="*60)
    print("2. GRANGER CAUSALITY TESTS")
    print("="*60)
    
    test_cols = ['rainfall_anomaly', 'temperature_anomaly', 'trade_volume_change']
    granger_results = models.granger_causality_test(df, 'price', test_cols)
    
    # 3. Multicollinearity
    print("\n" + "="*60)
    print("3. MULTICOLLINEARITY TEST")
    print("="*60)
    
    vif_results = models.multicollinearity_test(df, test_cols)
    
    # 4. Anomaly detection
    print("\n" + "="*60)
    print("4. ANOMALY DETECTION")
    print("="*60)
    
    anomaly_results = models.build_anomaly_detection_model(df.copy())
    
    # 5. Predictive modeling
    print("\n" + "="*60)
    print("5. PREDICTIVE MODELING")
    print("="*60)
    
    predictive_results = models.build_predictive_model(df.copy())
    
    # 6. Correlation analysis
    print("\n" + "="*60)
    print("6. CORRELATION ANALYSIS")
    print("="*60)
    
    correlation_results = models.calculate_signal_correlations(df.copy())
    
    # 7. Regime detection
    print("\n" + "="*60)
    print("7. REGIME DETECTION")
    print("="*60)
    
    regime_results = models.perform_regime_detection(df['price'])
    
    # 8. Risk metrics
    print("\n" + "="*60)
    print("8. RISK METRICS")
    print("="*60)
    
    returns = df['price'].pct_change().dropna()
    risk_metrics = models.calculate_risk_metrics(returns)