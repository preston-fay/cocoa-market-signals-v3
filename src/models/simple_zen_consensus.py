"""
Simple Zen Consensus for Price-Only Data
Works with real price data without requiring additional features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

class SimpleZenConsensus:
    """
    Simplified Zen Consensus that works with price data only
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Simple models that work with price only
        self.model_roles = {
            'trend_follower': {
                'models': ['sma', 'ema', 'linear_trend'],
                'stance': 'Follow the trend',
                'weight': 0.4
            },
            'mean_reverter': {
                'models': ['bollinger', 'rsi', 'zscore'],
                'stance': 'Expect reversion to mean',
                'weight': 0.3
            },
            'momentum_trader': {
                'models': ['momentum', 'macd', 'roc'],
                'stance': 'Trade on momentum',
                'weight': 0.3
            }
        }
        
        # Signal thresholds
        self.signal_thresholds = {
            'strong_buy': 0.02,    # >2% predicted increase
            'buy': 0.005,          # >0.5% increase
            'hold': -0.005,        # -0.5% to 0.5%
            'sell': -0.02,         # <-0.5% decrease
            'strong_sell': -0.05   # <-2% decrease
        }
    
    def calculate_sma(self, df: pd.DataFrame, period: int = 20) -> float:
        """Simple Moving Average prediction"""
        return df['price'].rolling(period).mean().iloc[-1]
    
    def calculate_ema(self, df: pd.DataFrame, period: int = 20) -> float:
        """Exponential Moving Average prediction"""
        return df['price'].ewm(span=period, adjust=False).mean().iloc[-1]
    
    def calculate_linear_trend(self, df: pd.DataFrame, period: int = 30) -> float:
        """Linear trend extrapolation"""
        recent = df.iloc[-period:]
        x = np.arange(len(recent))
        y = recent['price'].values
        
        # Fit linear regression
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        
        # Predict next value
        next_x = len(recent)
        return slope * next_x + intercept
    
    def calculate_bollinger_prediction(self, df: pd.DataFrame, period: int = 20) -> float:
        """Bollinger Bands mean reversion"""
        sma = df['price'].rolling(period).mean()
        std = df['price'].rolling(period).std()
        
        current_price = df['price'].iloc[-1]
        middle_band = sma.iloc[-1]
        upper_band = middle_band + 2 * std.iloc[-1]
        lower_band = middle_band - 2 * std.iloc[-1]
        
        # If price is near upper band, predict reversion down
        if current_price > middle_band + std.iloc[-1]:
            return middle_band
        # If price is near lower band, predict reversion up
        elif current_price < middle_band - std.iloc[-1]:
            return middle_band
        else:
            return current_price  # Within bands, no strong prediction
    
    def calculate_rsi_prediction(self, df: pd.DataFrame, period: int = 14) -> float:
        """RSI-based prediction"""
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        current_price = df['price'].iloc[-1]
        
        # Overbought (>70) - predict down
        if current_rsi > 70:
            return current_price * 0.98
        # Oversold (<30) - predict up
        elif current_rsi < 30:
            return current_price * 1.02
        else:
            return current_price
    
    def calculate_momentum(self, df: pd.DataFrame, period: int = 10) -> float:
        """Momentum-based prediction"""
        momentum = df['price'].iloc[-1] / df['price'].iloc[-period] - 1
        current_price = df['price'].iloc[-1]
        
        # Project momentum forward
        daily_momentum = momentum / period
        return current_price * (1 + daily_momentum * 7)  # 7-day forecast
    
    def run_model_role(self, df: pd.DataFrame, role: str) -> Dict[str, Any]:
        """Run all models for a specific role"""
        role_config = self.model_roles[role]
        predictions = {}
        
        for model in role_config['models']:
            try:
                if model == 'sma':
                    pred = self.calculate_sma(df)
                elif model == 'ema':
                    pred = self.calculate_ema(df)
                elif model == 'linear_trend':
                    pred = self.calculate_linear_trend(df)
                elif model == 'bollinger':
                    pred = self.calculate_bollinger_prediction(df)
                elif model == 'rsi':
                    pred = self.calculate_rsi_prediction(df)
                elif model == 'momentum':
                    pred = self.calculate_momentum(df)
                elif model == 'zscore':
                    # Z-score mean reversion
                    mean = df['price'].mean()
                    std = df['price'].std()
                    zscore = (df['price'].iloc[-1] - mean) / std
                    if abs(zscore) > 2:
                        pred = mean  # Predict reversion to mean
                    else:
                        pred = df['price'].iloc[-1]
                elif model == 'macd':
                    # Simple MACD trend
                    ema12 = df['price'].ewm(span=12, adjust=False).mean()
                    ema26 = df['price'].ewm(span=26, adjust=False).mean()
                    macd = ema12 - ema26
                    signal = macd.ewm(span=9, adjust=False).mean()
                    
                    if macd.iloc[-1] > signal.iloc[-1]:
                        pred = df['price'].iloc[-1] * 1.01  # Bullish
                    else:
                        pred = df['price'].iloc[-1] * 0.99  # Bearish
                elif model == 'roc':
                    # Rate of Change
                    roc = (df['price'].iloc[-1] / df['price'].iloc[-10] - 1) * 100
                    pred = df['price'].iloc[-1] * (1 + roc / 100 * 0.5)  # Project half the ROC
                else:
                    continue
                    
                predictions[model] = pred
                
            except Exception as e:
                self.logger.error(f"Error in {model}: {str(e)}")
        
        return {
            'role': role,
            'stance': role_config['stance'],
            'predictions': predictions,
            'weight': role_config['weight']
        }
    
    def run_consensus(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run full Zen Consensus"""
        current_price = df['price'].iloc[-1]
        role_results = []
        
        # Run each role
        for role in self.model_roles:
            result = self.run_model_role(df, role)
            role_results.append(result)
        
        # Calculate weighted consensus
        all_predictions = []
        all_weights = []
        
        for role_result in role_results:
            for model, pred in role_result['predictions'].items():
                if pred is not None:
                    all_predictions.append(pred)
                    all_weights.append(role_result['weight'] / len(role_result['predictions']))
        
        if not all_predictions:
            return {
                'consensus_forecast': current_price,
                'consensus_signal': 'hold',
                'confidence_score': 0.0,
                'error': 'No predictions generated'
            }
        
        # Weighted average
        consensus_price = np.average(all_predictions, weights=all_weights)
        
        # Calculate confidence based on agreement
        price_std = np.std(all_predictions)
        price_range = max(all_predictions) - min(all_predictions)
        agreement_score = 1 - (price_range / current_price / 0.1)  # 10% range = 0 confidence
        confidence = max(0, min(1, agreement_score))
        
        # Determine signal
        price_change = (consensus_price - current_price) / current_price
        
        if price_change > self.signal_thresholds['strong_buy']:
            signal = 'strong_buy'
        elif price_change > self.signal_thresholds['buy']:
            signal = 'buy'
        elif price_change > self.signal_thresholds['hold']:
            signal = 'hold'
        elif price_change > self.signal_thresholds['sell']:
            signal = 'sell'
        else:
            signal = 'strong_sell'
        
        # Generate reasoning
        reasoning = []
        for role_result in role_results:
            role_pred = np.mean(list(role_result['predictions'].values()))
            role_change = (role_pred - current_price) / current_price
            
            if abs(role_change) > 0.01:
                direction = "bullish" if role_change > 0 else "bearish"
                reasoning.append(f"{role_result['role']}: {direction} ({role_change*100:+.1f}%)")
        
        return {
            'timestamp': datetime.now(),
            'current_price': current_price,
            'consensus_forecast': consensus_price,
            'consensus_signal': signal,
            'confidence_score': confidence,
            'price_change_rate': price_change,
            'total_models': len(all_predictions),
            'role_contributions': {r['role']: r for r in role_results},
            'reasoning': reasoning,
            'prediction_range': {
                'min': min(all_predictions),
                'max': max(all_predictions),
                'std': price_std
            }
        }
    
    def get_market_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market context"""
        returns = df['price'].pct_change().dropna()
        current_vol = returns.rolling(20).std() * np.sqrt(252)
        avg_vol = returns.std() * np.sqrt(252)
        
        # Trend strength
        sma20 = df['price'].rolling(20).mean().iloc[-1]
        sma50 = df['price'].rolling(50).mean().iloc[-1]
        current_price = df['price'].iloc[-1]
        
        trend = 'bullish' if current_price > sma20 > sma50 else 'bearish' if current_price < sma20 < sma50 else 'neutral'
        
        return {
            'current_volatility': current_vol.iloc[-1],
            'avg_volatility': avg_vol,
            'volatility_regime': 'high' if current_vol.iloc[-1] > avg_vol * 1.5 else 'normal',
            'trend': trend,
            'price_vs_sma20': (current_price / sma20 - 1) * 100,
            'price_vs_sma50': (current_price / sma50 - 1) * 100
        }