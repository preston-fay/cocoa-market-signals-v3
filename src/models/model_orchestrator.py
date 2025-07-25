"""
Model Orchestrator for Dynamic Model Selection
Intelligently selects and combines models based on market regime
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

from .advanced_time_series_models import AdvancedTimeSeriesModels
from .statistical_models import StatisticalSignalModels

class ModelOrchestrator:
    """
    Orchestrates model selection based on market conditions
    Dynamically switches between models for optimal performance
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.advanced_models = AdvancedTimeSeriesModels()
        self.statistical_models = StatisticalSignalModels()
        
        # Model performance history
        self.performance_history = {}
        
        # Volatility thresholds
        self.volatility_thresholds = {
            'low': 30,      # Below 30th percentile
            'medium': 70,   # 30th to 70th percentile  
            'high': 100     # Above 70th percentile
        }
        
        # Model recommendations by regime
        self.regime_models = {
            'low_volatility': {
                'primary': ['xgboost', 'arima', 'holt_winters'],
                'secondary': ['sarima'],
                'anomaly': ['lof', 'modified_zscore']
            },
            'medium_volatility': {
                'primary': ['xgboost', 'sarima', 'ensemble'],
                'secondary': ['holt_winters'],
                'anomaly': ['cusum', 'lof']
            },
            'high_volatility': {
                'primary': ['lstm_predictor', 'ewma', 'xgboost'],
                'secondary': ['garch', 'ensemble'],
                'anomaly': ['lstm_autoencoder', 'modified_zscore']
            }
        }
        
    def detect_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect current market regime based on volatility and other factors
        """
        # Calculate EWMA volatility
        ewma_result = self.advanced_models.calculate_ewma(df)
        current_vol = ewma_result['current_vol']
        vol_percentile = ewma_result['vol_percentile']
        
        # Handle NaN percentile
        if np.isnan(vol_percentile):
            # Use GARCH if available
            try:
                garch_result = self.advanced_models.fit_garch(df)
                garch_vol = garch_result['conditional_volatility'].iloc[-1]
                high_vol_threshold = garch_result['conditional_volatility'].quantile(0.7)
                low_vol_threshold = garch_result['conditional_volatility'].quantile(0.3)
                
                if garch_vol > high_vol_threshold:
                    vol_percentile = 85
                    regime = 'high_volatility'
                elif garch_vol < low_vol_threshold:
                    vol_percentile = 15
                    regime = 'low_volatility'
                else:
                    vol_percentile = 50
                    regime = 'medium_volatility'
            except:
                # Default to medium
                vol_percentile = 50
                regime = 'medium_volatility'
        else:
            # Determine regime based on percentile
            if vol_percentile < self.volatility_thresholds['low']:
                regime = 'low_volatility'
            elif vol_percentile < self.volatility_thresholds['medium']:
                regime = 'medium_volatility'
            else:
                regime = 'high_volatility'
        
        # Additional regime indicators
        price_series = df['price']
        
        # Recent price momentum
        momentum_5d = (price_series.iloc[-1] / price_series.iloc[-5] - 1) * 100
        momentum_20d = (price_series.iloc[-1] / price_series.iloc[-20] - 1) * 100
        
        # Volume changes
        if 'trade_volume_change' in df.columns:
            recent_volume = df['trade_volume_change'].iloc[-20:].mean()
        else:
            recent_volume = 0
        
        return {
            'regime': regime,
            'volatility': current_vol,
            'volatility_percentile': vol_percentile,
            'momentum_5d': momentum_5d,
            'momentum_20d': momentum_20d,
            'volume_activity': recent_volume,
            'timestamp': datetime.now()
        }
    
    def select_models(self, regime: str, task: str = 'forecast') -> List[str]:
        """
        Select appropriate models based on regime and task
        """
        if task == 'forecast':
            return self.regime_models[regime]['primary']
        elif task == 'anomaly':
            return self.regime_models[regime]['anomaly']
        else:
            return self.regime_models[regime]['primary'] + self.regime_models[regime]['secondary']
    
    def run_orchestrated_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run full orchestrated analysis with dynamic model selection
        """
        self.logger.info("Starting orchestrated analysis...")
        
        # Prepare data
        df = self.advanced_models.prepare_data(df)
        
        # 1. Detect market regime
        regime_info = self.detect_market_regime(df)
        regime = regime_info['regime']
        
        self.logger.info(f"Detected market regime: {regime}")
        self.logger.info(f"Volatility: {regime_info['volatility']:.2f}% (percentile: {regime_info['volatility_percentile']:.0f}%)")
        
        # 2. Select models for current regime
        forecast_models = self.select_models(regime, 'forecast')
        anomaly_models = self.select_models(regime, 'anomaly')
        
        results = {
            'regime': regime_info,
            'models_used': {
                'forecast': forecast_models,
                'anomaly': anomaly_models
            },
            'forecasts': {},
            'anomalies': {},
            'signals': {},
            'recommendations': []
        }
        
        # 3. Run forecast models
        self.logger.info(f"Running forecast models: {forecast_models}")
        
        for model_name in forecast_models:
            try:
                if model_name == 'xgboost':
                    model_result = self.advanced_models.fit_xgboost(df)
                elif model_name == 'arima':
                    model_result = self.advanced_models.fit_arima(df)
                elif model_name == 'sarima':
                    model_result = self.advanced_models.fit_sarima(df)
                elif model_name == 'holt_winters':
                    model_result = self.advanced_models.fit_holt_winters(df)
                elif model_name == 'garch':
                    model_result = self.advanced_models.fit_garch(df)
                elif model_name == 'lstm_predictor':
                    model_result = self.advanced_models.fit_lstm_predictor(df)
                elif model_name == 'ewma':
                    model_result = self.advanced_models.calculate_ewma(df)
                elif model_name == 'ensemble':
                    # Run base models first
                    self.advanced_models.fit_arima(df)
                    self.advanced_models.fit_sarima(df)
                    self.advanced_models.fit_holt_winters(df)
                    model_result = self.advanced_models.create_ensemble_predictions(df)
                
                if model_result and 'forecast' in model_result:
                    results['forecasts'][model_name] = model_result
                    
            except Exception as e:
                self.logger.error(f"Error running {model_name}: {str(e)}")
        
        # 4. Run anomaly detection models
        self.logger.info(f"Running anomaly models: {anomaly_models}")
        
        for model_name in anomaly_models:
            try:
                if model_name == 'lof':
                    model_result = self.advanced_models.detect_lof(df)
                elif model_name == 'modified_zscore':
                    model_result = self.advanced_models.detect_modified_zscore(df)
                elif model_name == 'cusum':
                    model_result = self.advanced_models.detect_cusum(df)
                elif model_name == 'lstm_autoencoder':
                    model_result = self.advanced_models.fit_lstm_autoencoder(df)
                
                if model_result:
                    results['anomalies'][model_name] = model_result
                    
            except Exception as e:
                self.logger.error(f"Error running {model_name}: {str(e)}")
        
        # 5. Generate trading signals
        results['signals'] = self.generate_trading_signals(results, df)
        
        # 6. Generate recommendations
        results['recommendations'] = self.generate_recommendations(results, regime_info)
        
        return results
    
    def generate_trading_signals(self, results: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate combined trading signals from multiple models
        """
        signals = {
            'timestamp': datetime.now(),
            'price_forecast': None,
            'confidence': 0,
            'signal_strength': 0,
            'direction': 'neutral',
            'risk_level': 'medium'
        }
        
        # Combine forecasts with weights based on historical performance
        if results['forecasts']:
            # For now, use simple average (in production, use performance-weighted)
            forecasts = []
            weights = []
            
            for model_name, model_result in results['forecasts'].items():
                if 'forecast' in model_result:
                    forecast_data = model_result['forecast']
                    
                    # Handle different forecast formats
                    if hasattr(forecast_data, 'variance'):  # GARCH forecast object
                        # Get mean forecast
                        forecast_values = forecast_data.mean.iloc[-1, :]
                        forecast_value = forecast_values.iloc[6] if len(forecast_values) > 6 else forecast_values.iloc[-1]
                    elif hasattr(forecast_data, 'iloc'):  # Pandas Series/DataFrame
                        forecast_value = forecast_data.iloc[6] if len(forecast_data) > 6 else forecast_data.iloc[-1]
                    elif isinstance(forecast_data, (list, np.ndarray)):  # List or array
                        forecast_value = forecast_data[6] if len(forecast_data) > 6 else forecast_data[-1]
                    else:
                        continue  # Skip if format unknown
                    
                    forecasts.append(forecast_value)
                    
                    # Weight based on model performance (XGBoost gets highest weight)
                    if model_name == 'xgboost':
                        weights.append(0.4)
                    elif model_name in ['arima', 'holt_winters']:
                        weights.append(0.2)
                    else:
                        weights.append(0.1)
            
            if forecasts:
                # Normalize weights
                weights = np.array(weights) / np.sum(weights)
                
                # Weighted average forecast
                signals['price_forecast'] = np.average(forecasts, weights=weights)
                
                # Calculate signal strength and direction
                current_price = df['price'].iloc[-1]
                price_change_pct = (signals['price_forecast'] - current_price) / current_price * 100
                
                if price_change_pct > 2:
                    signals['direction'] = 'bullish'
                    signals['signal_strength'] = min(abs(price_change_pct) / 10, 1.0)
                elif price_change_pct < -2:
                    signals['direction'] = 'bearish'
                    signals['signal_strength'] = min(abs(price_change_pct) / 10, 1.0)
                else:
                    signals['direction'] = 'neutral'
                    signals['signal_strength'] = 0.2
                
                # Confidence based on model agreement
                forecast_std = np.std(forecasts)
                signals['confidence'] = max(0, 1 - (forecast_std / current_price))
        
        # Adjust risk level based on anomalies
        total_anomalies = sum(
            len(result.get('anomaly_dates', [])) 
            for result in results['anomalies'].values()
        )
        
        if total_anomalies > 10:
            signals['risk_level'] = 'high'
        elif total_anomalies > 5:
            signals['risk_level'] = 'medium'
        else:
            signals['risk_level'] = 'low'
        
        return signals
    
    def generate_recommendations(self, results: Dict, regime_info: Dict) -> List[str]:
        """
        Generate actionable recommendations based on analysis
        """
        recommendations = []
        
        # Regime-based recommendations
        if regime_info['regime'] == 'high_volatility':
            recommendations.append("⚠️ High volatility detected - Consider reducing position sizes")
            recommendations.append("📊 Monitor GARCH volatility forecasts closely")
            recommendations.append("🛡️ Implement stop-loss orders at -5% from entry")
        elif regime_info['regime'] == 'low_volatility':
            recommendations.append("✅ Stable market conditions - Good for trend-following strategies")
            recommendations.append("📈 XGBoost showing high confidence in predictions")
            recommendations.append("🎯 Consider mean-reversion strategies")
        
        # Signal-based recommendations
        signals = results['signals']
        if signals['direction'] == 'bullish' and signals['confidence'] > 0.7:
            recommendations.append(f"🚀 Strong BUY signal - Target: ${signals['price_forecast']:,.0f}")
        elif signals['direction'] == 'bearish' and signals['confidence'] > 0.7:
            recommendations.append(f"📉 Strong SELL signal - Target: ${signals['price_forecast']:,.0f}")
        
        # Anomaly-based recommendations
        if results['anomalies']:
            for model_name, anomaly_result in results['anomalies'].items():
                if 'anomaly_dates' in anomaly_result and len(anomaly_result['anomaly_dates']) > 0:
                    recent_anomalies = [
                        date for date in anomaly_result['anomaly_dates'] 
                        if (pd.Timestamp.now() - pd.Timestamp(date)).days < 7
                    ]
                    if recent_anomalies:
                        recommendations.append(f"🔍 {model_name.upper()} detected {len(recent_anomalies)} anomalies in past week")
        
        # Momentum-based recommendations
        if regime_info['momentum_5d'] > 5:
            recommendations.append("📊 Strong 5-day momentum (+{:.1f}%) - Trend continuation likely".format(regime_info['momentum_5d']))
        elif regime_info['momentum_5d'] < -5:
            recommendations.append("📉 Negative 5-day momentum ({:.1f}%) - Consider defensive positions".format(regime_info['momentum_5d']))
        
        return recommendations
    
    def backtest_orchestration(self, df: pd.DataFrame, test_period_days: int = 90) -> Dict[str, Any]:
        """
        Backtest the orchestration strategy
        """
        # Split data
        test_start_idx = len(df) - test_period_days
        
        results = {
            'dates': [],
            'actual_prices': [],
            'predicted_prices': [],
            'regimes': [],
            'signals': [],
            'returns': []
        }
        
        # Run rolling window backtest
        for i in range(test_start_idx, len(df) - 7):  # -7 for forecast horizon
            # Use data up to current point
            current_data = df.iloc[:i]
            
            # Run orchestrated analysis
            analysis = self.run_orchestrated_analysis(current_data)
            
            # Record results
            results['dates'].append(df.index[i])
            results['actual_prices'].append(df['price'].iloc[i])
            results['predicted_prices'].append(analysis['signals']['price_forecast'])
            results['regimes'].append(analysis['regime']['regime'])
            results['signals'].append(analysis['signals']['direction'])
            
            # Calculate returns if following signals
            if i > test_start_idx:
                actual_return = (df['price'].iloc[i] - df['price'].iloc[i-1]) / df['price'].iloc[i-1]
                if results['signals'][-2] == 'bullish':
                    results['returns'].append(actual_return)
                elif results['signals'][-2] == 'bearish':
                    results['returns'].append(-actual_return)
                else:
                    results['returns'].append(0)
        
        # Calculate performance metrics
        results_df = pd.DataFrame(results)
        
        # Prediction accuracy
        results_df['prediction_error'] = results_df['predicted_prices'] - results_df['actual_prices']
        mape = np.mean(np.abs(results_df['prediction_error'] / results_df['actual_prices'])) * 100
        
        # Trading performance
        cumulative_returns = (1 + pd.Series(results['returns'])).cumprod() - 1
        sharpe_ratio = np.sqrt(252) * np.mean(results['returns']) / np.std(results['returns']) if np.std(results['returns']) > 0 else 0
        
        return {
            'results_df': results_df,
            'mape': mape,
            'cumulative_returns': cumulative_returns.iloc[-1] * 100 if len(cumulative_returns) > 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'regime_distribution': results_df['regimes'].value_counts().to_dict()
        }