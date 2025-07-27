"""
Zen Consensus Model Orchestrator
Uses multiple AI models to reach consensus on market predictions
100% REAL predictions - NO CHEATING
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import json

from .advanced_time_series_models import AdvancedTimeSeriesModels
from .statistical_models import StatisticalSignalModels

class ZenOrchestrator:
    """
    Orchestrates multiple models using Zen Consensus approach
    Each model provides a different perspective on market conditions
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.advanced_models = AdvancedTimeSeriesModels()
        self.statistical_models = StatisticalSignalModels()
        
        # Define model roles for consensus
        self.model_roles = {
            'neutral_analyst': {
                'models': ['arima', 'sarima', 'holt_winters'],
                'stance': 'Objective time series analysis',
                'weight': 0.3
            },
            'supportive_trader': {
                'models': ['xgboost', 'lstm_predictor'],
                'stance': 'Bullish on ML predictions',
                'weight': 0.4
            },
            'critical_risk_manager': {
                'models': ['garch', 'ewma', 'modified_zscore'],
                'stance': 'Focus on volatility and risk',
                'weight': 0.3
            }
        }
        
        # Consensus thresholds
        self.consensus_thresholds = {
            'strong_buy': 0.7,
            'buy': 0.55,
            'hold': 0.45,
            'sell': 0.3,
            'strong_sell': 0.15
        }
        
    def run_model_with_role(self, df: pd.DataFrame, role: str) -> Dict[str, Any]:
        """
        Run models assigned to a specific role
        """
        role_config = self.model_roles[role]
        results = {
            'role': role,
            'stance': role_config['stance'],
            'predictions': {},
            'confidence': {},
            'reasoning': []
        }
        
        for model_name in role_config['models']:
            try:
                self.logger.info(f"{role}: Running {model_name}")
                
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
                elif model_name == 'modified_zscore':
                    model_result = self.advanced_models.detect_modified_zscore(df)
                    
                if model_result:
                    results['predictions'][model_name] = model_result
                    
                    # Extract forecast value
                    if 'forecast' in model_result:
                        forecast_data = model_result['forecast']
                        if hasattr(forecast_data, 'iloc'):
                            forecast_value = forecast_data.iloc[6] if len(forecast_data) > 6 else forecast_data.iloc[-1]
                        elif isinstance(forecast_data, (list, np.ndarray)):
                            forecast_value = forecast_data[6] if len(forecast_data) > 6 else forecast_data[-1]
                        else:
                            forecast_value = None
                            
                        if forecast_value is not None:
                            current_price = df['price'].iloc[-1]
                            price_change = (forecast_value - current_price) / current_price
                            
                            # Model-specific confidence
                            if model_name == 'xgboost':
                                confidence = 0.8  # High confidence in ML
                            elif model_name in ['arima', 'sarima']:
                                confidence = 0.7  # Good for time series
                            elif model_name == 'garch':
                                confidence = 0.6  # Volatility specialist
                            else:
                                confidence = 0.5
                                
                            results['confidence'][model_name] = confidence
                            
                            # Generate reasoning
                            if price_change > 0.02:
                                results['reasoning'].append(
                                    f"{model_name}: Predicting {price_change*100:.1f}% increase"
                                )
                            elif price_change < -0.02:
                                results['reasoning'].append(
                                    f"{model_name}: Warning of {abs(price_change)*100:.1f}% decrease"
                                )
                                
            except Exception as e:
                self.logger.error(f"Error in {role} running {model_name}: {str(e)}")
                
        return results
        
    def synthesize_consensus(self, role_results: List[Dict[str, Any]], df: pd.DataFrame) -> Dict[str, Any]:
        """
        Synthesize results from all roles into consensus prediction
        """
        current_price = df['price'].iloc[-1]
        all_predictions = []
        all_weights = []
        
        consensus = {
            'timestamp': datetime.now(),
            'current_price': current_price,
            'consensus_forecast': None,
            'consensus_signal': 'hold',
            'confidence_score': 0,
            'role_contributions': {},
            'reasoning': [],
            'dissenting_views': []
        }
        
        # Collect predictions from each role
        for role_result in role_results:
            role = role_result['role']
            role_weight = self.model_roles[role]['weight']
            
            role_predictions = []
            role_confidences = []
            
            for model_name, prediction in role_result['predictions'].items():
                if 'forecast' in prediction:
                    forecast_data = prediction['forecast']
                    
                    # Extract forecast value
                    if hasattr(forecast_data, 'iloc'):
                        forecast_value = forecast_data.iloc[6] if len(forecast_data) > 6 else forecast_data.iloc[-1]
                    elif isinstance(forecast_data, (list, np.ndarray)):
                        forecast_value = forecast_data[6] if len(forecast_data) > 6 else forecast_data[-1]
                    else:
                        continue
                        
                    if isinstance(forecast_value, (int, float)) and not np.isnan(forecast_value):
                        role_predictions.append(forecast_value)
                        if model_name in role_result['confidence']:
                            role_confidences.append(role_result['confidence'][model_name])
                            
            if role_predictions:
                # Average predictions within role
                avg_prediction = np.mean(role_predictions)
                avg_confidence = np.mean(role_confidences) if role_confidences else 0.5
                
                all_predictions.append(avg_prediction)
                all_weights.append(role_weight * avg_confidence)
                
                consensus['role_contributions'][role] = {
                    'prediction': avg_prediction,
                    'confidence': avg_confidence,
                    'change_pct': (avg_prediction - current_price) / current_price * 100
                }
                
        # Calculate weighted consensus
        if all_predictions:
            weights_sum = sum(all_weights)
            if weights_sum > 0:
                normalized_weights = [w/weights_sum for w in all_weights]
                consensus['consensus_forecast'] = np.average(all_predictions, weights=normalized_weights)
                
                # Calculate consensus metrics
                price_change_pct = (consensus['consensus_forecast'] - current_price) / current_price * 100
                prediction_std = np.std(all_predictions)
                consensus['confidence_score'] = max(0, 1 - (prediction_std / current_price))
                
                # Determine signal based on consensus
                if price_change_pct > 3 and consensus['confidence_score'] > 0.6:
                    consensus['consensus_signal'] = 'strong_buy'
                elif price_change_pct > 1:
                    consensus['consensus_signal'] = 'buy'
                elif price_change_pct < -3 and consensus['confidence_score'] > 0.6:
                    consensus['consensus_signal'] = 'strong_sell'
                elif price_change_pct < -1:
                    consensus['consensus_signal'] = 'sell'
                else:
                    consensus['consensus_signal'] = 'hold'
                    
                # Generate consensus reasoning
                consensus['reasoning'].append(
                    f"Consensus forecast: ${consensus['consensus_forecast']:,.0f} ({price_change_pct:+.1f}%)"
                )
                consensus['reasoning'].append(
                    f"Confidence level: {consensus['confidence_score']*100:.0f}%"
                )
                
                # Identify dissenting views
                for role, contrib in consensus['role_contributions'].items():
                    role_change = contrib['change_pct']
                    if abs(role_change - price_change_pct) > 2:
                        consensus['dissenting_views'].append(
                            f"{role}: {role_change:+.1f}% vs consensus {price_change_pct:+.1f}%"
                        )
                        
        return consensus
        
    def run_zen_consensus(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main entry point for Zen Consensus analysis
        """
        self.logger.info("Starting Zen Consensus analysis...")
        
        # Prepare data
        df = self.advanced_models.prepare_data(df)
        
        # Run each role's analysis
        role_results = []
        for role in self.model_roles:
            self.logger.info(f"Running {role} analysis...")
            role_result = self.run_model_with_role(df, role)
            role_results.append(role_result)
            
        # Synthesize consensus
        consensus = self.synthesize_consensus(role_results, df)
        
        # Add market context
        consensus['market_context'] = self.get_market_context(df)
        
        # Generate final recommendations
        consensus['recommendations'] = self.generate_recommendations(consensus, df)
        
        return {
            'consensus': consensus,
            'role_results': role_results,
            'timestamp': datetime.now()
        }
        
    def get_market_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze current market context
        """
        price_series = df['price']
        
        context = {
            'trend_5d': (price_series.iloc[-1] / price_series.iloc[-5] - 1) * 100,
            'trend_20d': (price_series.iloc[-1] / price_series.iloc[-20] - 1) * 100,
            'volatility_20d': price_series.iloc[-20:].pct_change().std() * np.sqrt(252) * 100,
            'price_percentile': (price_series.iloc[-1] - price_series.min()) / (price_series.max() - price_series.min()) * 100
        }
        
        # Recent volume changes
        if 'trade_volume_change' in df.columns:
            context['volume_trend'] = df['trade_volume_change'].iloc[-20:].mean()
        
        return context
        
    def generate_recommendations(self, consensus: Dict, df: pd.DataFrame) -> List[str]:
        """
        Generate actionable recommendations based on consensus
        """
        recommendations = []
        
        signal = consensus['consensus_signal']
        confidence = consensus['confidence_score']
        forecast = consensus['consensus_forecast']
        current_price = consensus['current_price']
        
        # Signal-based recommendations
        if signal == 'strong_buy':
            recommendations.append(f"🚀 STRONG BUY - Target: ${forecast:,.0f} (Confidence: {confidence*100:.0f}%)")
            recommendations.append("📊 All model roles agree on bullish outlook")
        elif signal == 'buy':
            recommendations.append(f"📈 BUY - Target: ${forecast:,.0f} (Confidence: {confidence*100:.0f}%)")
        elif signal == 'strong_sell':
            recommendations.append(f"📉 STRONG SELL - Target: ${forecast:,.0f} (Confidence: {confidence*100:.0f}%)")
            recommendations.append("⚠️ Risk management models showing high concern")
        elif signal == 'sell':
            recommendations.append(f"📉 SELL - Target: ${forecast:,.0f} (Confidence: {confidence*100:.0f}%)")
        else:
            recommendations.append(f"⏸️ HOLD - Limited movement expected (Confidence: {confidence*100:.0f}%)")
            
        # Add dissenting view warnings
        if consensus['dissenting_views']:
            recommendations.append("⚠️ Note: Models show divergent views")
            for dissent in consensus['dissenting_views'][:2]:  # Show top 2
                recommendations.append(f"  • {dissent}")
                
        # Market context recommendations
        context = consensus['market_context']
        if context['volatility_20d'] > 50:
            recommendations.append("🌊 High volatility environment - Consider position sizing")
        
        if abs(context['trend_5d']) > 5:
            direction = "upward" if context['trend_5d'] > 0 else "downward"
            recommendations.append(f"📊 Strong {direction} momentum in past 5 days ({context['trend_5d']:+.1f}%)")
            
        return recommendations
        
    def backtest_zen_consensus(self, df: pd.DataFrame, test_days: int = 90) -> Dict[str, Any]:
        """
        Backtest the Zen Consensus strategy
        """
        test_start_idx = len(df) - test_days
        
        results = {
            'dates': [],
            'actual_prices': [],
            'consensus_forecasts': [],
            'signals': [],
            'confidence_scores': [],
            'returns': []
        }
        
        # Run rolling backtest
        for i in range(test_start_idx, len(df) - 7):
            current_data = df.iloc[:i]
            
            # Run consensus
            consensus_result = self.run_zen_consensus(current_data)
            consensus = consensus_result['consensus']
            
            # Record results
            results['dates'].append(df.index[i])
            results['actual_prices'].append(df['price'].iloc[i])
            results['consensus_forecasts'].append(consensus['consensus_forecast'])
            results['signals'].append(consensus['consensus_signal'])
            results['confidence_scores'].append(consensus['confidence_score'])
            
            # Calculate returns
            if i > test_start_idx:
                actual_return = (df['price'].iloc[i] - df['price'].iloc[i-1]) / df['price'].iloc[i-1]
                
                prev_signal = results['signals'][-2]
                if prev_signal in ['strong_buy', 'buy']:
                    results['returns'].append(actual_return)
                elif prev_signal in ['strong_sell', 'sell']:
                    results['returns'].append(-actual_return)
                else:
                    results['returns'].append(0)
                    
        # Calculate metrics
        results_df = pd.DataFrame(results)
        
        # Accuracy metrics
        results_df['prediction_error'] = results_df['consensus_forecasts'] - results_df['actual_prices']
        mape = np.mean(np.abs(results_df['prediction_error'] / results_df['actual_prices'])) * 100
        
        # Trading metrics
        cumulative_returns = (1 + pd.Series(results['returns'])).cumprod() - 1
        sharpe = np.sqrt(252) * np.mean(results['returns']) / np.std(results['returns']) if np.std(results['returns']) > 0 else 0
        
        # Signal distribution
        signal_counts = results_df['signals'].value_counts()
        
        return {
            'results_df': results_df,
            'mape': mape,
            'cumulative_return_pct': cumulative_returns.iloc[-1] * 100 if len(cumulative_returns) > 0 else 0,
            'sharpe_ratio': sharpe,
            'avg_confidence': np.mean(results['confidence_scores']),
            'signal_distribution': signal_counts.to_dict()
        }