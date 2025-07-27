"""
Multi-Source Signal Detection System
Detects honest market signals from multiple REAL data sources
NO FAKE SIGNALS - 100% data-driven
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from scipy import stats

class MultiSourceSignalDetector:
    """
    Detects market signals from multiple real data sources:
    - Price movements (Yahoo Finance)
    - Trade volume changes (UN Comtrade)
    - Weather patterns (Open-Meteo)
    - Technical indicators
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Signal thresholds
        self.signal_thresholds = {
            'price': {
                'strong_bullish': 0.05,  # 5% move
                'bullish': 0.02,         # 2% move
                'bearish': -0.02,        # -2% move
                'strong_bearish': -0.05  # -5% move
            },
            'volume': {
                'surge': 1.5,      # 50% above average
                'high': 1.2,       # 20% above average
                'low': 0.8,        # 20% below average
                'drought': 0.5     # 50% below average
            },
            'weather': {
                'severe_drought': -2.0,  # Z-score
                'drought': -1.5,
                'normal': (-1.5, 1.5),
                'excess_rain': 1.5,
                'flood': 2.0
            },
            'technical': {
                'oversold': 30,
                'neutral': (30, 70),
                'overbought': 70
            }
        }
        
    def detect_price_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect signals from price movements"""
        signals = {
            'timestamp': datetime.now(),
            'current_price': df['price'].iloc[-1],
            'signals': [],
            'strength': 0
        }
        
        # Short-term momentum (5 days)
        momentum_5d = (df['price'].iloc[-1] / df['price'].iloc[-5] - 1)
        
        # Medium-term momentum (20 days)
        momentum_20d = (df['price'].iloc[-1] / df['price'].iloc[-20] - 1)
        
        # Moving average crossovers
        ma_20 = df['price'].rolling(20).mean()
        ma_50 = df['price'].rolling(50).mean()
        
        current_price = df['price'].iloc[-1]
        
        # Price momentum signals
        if momentum_5d > self.signal_thresholds['price']['strong_bullish']:
            signals['signals'].append({
                'type': 'price_momentum',
                'signal': 'strong_bullish',
                'value': momentum_5d,
                'description': f'Strong 5-day momentum: {momentum_5d*100:.1f}%'
            })
            signals['strength'] += 2
        elif momentum_5d > self.signal_thresholds['price']['bullish']:
            signals['signals'].append({
                'type': 'price_momentum',
                'signal': 'bullish',
                'value': momentum_5d,
                'description': f'Positive 5-day momentum: {momentum_5d*100:.1f}%'
            })
            signals['strength'] += 1
        elif momentum_5d < self.signal_thresholds['price']['strong_bearish']:
            signals['signals'].append({
                'type': 'price_momentum',
                'signal': 'strong_bearish',
                'value': momentum_5d,
                'description': f'Strong negative 5-day momentum: {momentum_5d*100:.1f}%'
            })
            signals['strength'] -= 2
        elif momentum_5d < self.signal_thresholds['price']['bearish']:
            signals['signals'].append({
                'type': 'price_momentum',
                'signal': 'bearish',
                'value': momentum_5d,
                'description': f'Negative 5-day momentum: {momentum_5d*100:.1f}%'
            })
            signals['strength'] -= 1
            
        # Moving average signals
        if len(ma_20) > 1 and len(ma_50) > 1:
            if ma_20.iloc[-1] > ma_50.iloc[-1] and ma_20.iloc[-2] <= ma_50.iloc[-2]:
                signals['signals'].append({
                    'type': 'ma_crossover',
                    'signal': 'golden_cross',
                    'value': ma_20.iloc[-1] / ma_50.iloc[-1],
                    'description': 'MA20 crossed above MA50 (Golden Cross)'
                })
                signals['strength'] += 2
            elif ma_20.iloc[-1] < ma_50.iloc[-1] and ma_20.iloc[-2] >= ma_50.iloc[-2]:
                signals['signals'].append({
                    'type': 'ma_crossover',
                    'signal': 'death_cross',
                    'value': ma_20.iloc[-1] / ma_50.iloc[-1],
                    'description': 'MA20 crossed below MA50 (Death Cross)'
                })
                signals['strength'] -= 2
                
        # Price level signals
        price_52w_high = df['price'].iloc[-252:].max() if len(df) > 252 else df['price'].max()
        price_52w_low = df['price'].iloc[-252:].min() if len(df) > 252 else df['price'].min()
        
        if current_price > price_52w_high * 0.95:
            signals['signals'].append({
                'type': 'price_level',
                'signal': 'near_52w_high',
                'value': current_price / price_52w_high,
                'description': f'Price near 52-week high (${price_52w_high:.0f})'
            })
            signals['strength'] += 1
        elif current_price < price_52w_low * 1.05:
            signals['signals'].append({
                'type': 'price_level',
                'signal': 'near_52w_low',
                'value': current_price / price_52w_low,
                'description': f'Price near 52-week low (${price_52w_low:.0f})'
            })
            signals['strength'] -= 1
            
        return signals
        
    def detect_volume_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect signals from trade volume changes"""
        signals = {
            'timestamp': datetime.now(),
            'signals': [],
            'strength': 0
        }
        
        if 'trade_volume_change' not in df.columns:
            return signals
            
        # Recent volume activity
        recent_volume = df['trade_volume_change'].iloc[-20:]
        avg_volume = recent_volume.mean()
        volume_std = recent_volume.std()
        current_volume = df['trade_volume_change'].iloc[-1]
        
        # Volume relative to average
        if avg_volume != 0:
            volume_ratio = (current_volume + 1) / (avg_volume + 1)
        else:
            volume_ratio = 1.0
            
        # Volume surge/drought signals
        if volume_ratio > self.signal_thresholds['volume']['surge']:
            signals['signals'].append({
                'type': 'volume_anomaly',
                'signal': 'volume_surge',
                'value': volume_ratio,
                'description': f'Volume surge: {volume_ratio:.1f}x average'
            })
            signals['strength'] += 2
        elif volume_ratio > self.signal_thresholds['volume']['high']:
            signals['signals'].append({
                'type': 'volume_trend',
                'signal': 'high_volume',
                'value': volume_ratio,
                'description': f'Above average volume: {volume_ratio:.1f}x'
            })
            signals['strength'] += 1
        elif volume_ratio < self.signal_thresholds['volume']['drought']:
            signals['signals'].append({
                'type': 'volume_anomaly',
                'signal': 'volume_drought',
                'value': volume_ratio,
                'description': f'Volume drought: {volume_ratio:.1f}x average'
            })
            signals['strength'] -= 1
            
        # Volume trend signals
        volume_trend = recent_volume.rolling(5).mean()
        if len(volume_trend) > 1:
            trend_direction = (volume_trend.iloc[-1] - volume_trend.iloc[-5]) / abs(volume_trend.iloc[-5] + 0.001)
            
            if trend_direction > 0.2:
                signals['signals'].append({
                    'type': 'volume_trend',
                    'signal': 'increasing_volume',
                    'value': trend_direction,
                    'description': f'Volume increasing: {trend_direction*100:.0f}% over 5 days'
                })
                signals['strength'] += 1
            elif trend_direction < -0.2:
                signals['signals'].append({
                    'type': 'volume_trend',
                    'signal': 'decreasing_volume',
                    'value': trend_direction,
                    'description': f'Volume decreasing: {abs(trend_direction)*100:.0f}% over 5 days'
                })
                signals['strength'] -= 1
                
        return signals
        
    def detect_weather_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect signals from weather patterns"""
        signals = {
            'timestamp': datetime.now(),
            'signals': [],
            'strength': 0
        }
        
        # Check for weather columns
        weather_cols = [col for col in df.columns if 'precip' in col or 'temp' in col]
        if not weather_cols:
            return signals
            
        # Analyze precipitation patterns
        if 'avg_precipitation' in df.columns:
            recent_precip = df['avg_precipitation'].iloc[-30:]
            historical_precip = df['avg_precipitation'].iloc[:-30]
            
            # Calculate z-score
            if len(historical_precip) > 30:
                precip_mean = historical_precip.mean()
                precip_std = historical_precip.std()
                
                if precip_std > 0:
                    current_z = (recent_precip.mean() - precip_mean) / precip_std
                    
                    if current_z < self.signal_thresholds['weather']['severe_drought']:
                        signals['signals'].append({
                            'type': 'weather_extreme',
                            'signal': 'severe_drought',
                            'value': current_z,
                            'description': f'Severe drought conditions (Z-score: {current_z:.2f})'
                        })
                        signals['strength'] -= 2
                    elif current_z < self.signal_thresholds['weather']['drought']:
                        signals['signals'].append({
                            'type': 'weather_anomaly',
                            'signal': 'drought',
                            'value': current_z,
                            'description': f'Drought conditions (Z-score: {current_z:.2f})'
                        })
                        signals['strength'] -= 1
                    elif current_z > self.signal_thresholds['weather']['flood']:
                        signals['signals'].append({
                            'type': 'weather_extreme',
                            'signal': 'flood_risk',
                            'value': current_z,
                            'description': f'Flood risk conditions (Z-score: {current_z:.2f})'
                        })
                        signals['strength'] -= 2
                    elif current_z > self.signal_thresholds['weather']['excess_rain']:
                        signals['signals'].append({
                            'type': 'weather_anomaly',
                            'signal': 'excess_rain',
                            'value': current_z,
                            'description': f'Excess rainfall (Z-score: {current_z:.2f})'
                        })
                        signals['strength'] -= 1
                        
        # Temperature anomalies
        if 'avg_temperature' in df.columns:
            recent_temp = df['avg_temperature'].iloc[-30:]
            historical_temp = df['avg_temperature'].iloc[:-30]
            
            if len(historical_temp) > 30:
                temp_mean = historical_temp.mean()
                temp_std = historical_temp.std()
                
                if temp_std > 0:
                    temp_z = (recent_temp.mean() - temp_mean) / temp_std
                    
                    if abs(temp_z) > 2:
                        signals['signals'].append({
                            'type': 'weather_anomaly',
                            'signal': 'temperature_extreme',
                            'value': temp_z,
                            'description': f'Extreme temperature anomaly (Z-score: {temp_z:.2f})'
                        })
                        signals['strength'] -= 1
                        
        return signals
        
    def detect_technical_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect technical analysis signals"""
        signals = {
            'timestamp': datetime.now(),
            'signals': [],
            'strength': 0
        }
        
        prices = df['price']
        
        # RSI (Relative Strength Index)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        if not np.isnan(current_rsi):
            if current_rsi < self.signal_thresholds['technical']['oversold']:
                signals['signals'].append({
                    'type': 'technical',
                    'signal': 'oversold',
                    'value': current_rsi,
                    'description': f'RSI oversold: {current_rsi:.0f}'
                })
                signals['strength'] += 1
            elif current_rsi > self.signal_thresholds['technical']['overbought']:
                signals['signals'].append({
                    'type': 'technical',
                    'signal': 'overbought',
                    'value': current_rsi,
                    'description': f'RSI overbought: {current_rsi:.0f}'
                })
                signals['strength'] -= 1
                
        # Bollinger Bands
        sma_20 = prices.rolling(20).mean()
        std_20 = prices.rolling(20).std()
        upper_band = sma_20 + (2 * std_20)
        lower_band = sma_20 - (2 * std_20)
        
        current_price = prices.iloc[-1]
        
        if len(upper_band) > 0 and not np.isnan(upper_band.iloc[-1]):
            if current_price > upper_band.iloc[-1]:
                signals['signals'].append({
                    'type': 'technical',
                    'signal': 'above_upper_band',
                    'value': current_price / upper_band.iloc[-1],
                    'description': 'Price above upper Bollinger Band'
                })
                signals['strength'] -= 1
            elif current_price < lower_band.iloc[-1]:
                signals['signals'].append({
                    'type': 'technical',
                    'signal': 'below_lower_band',
                    'value': current_price / lower_band.iloc[-1],
                    'description': 'Price below lower Bollinger Band'
                })
                signals['strength'] += 1
                
        # MACD
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9, adjust=False).mean()
        
        if len(macd) > 1 and len(signal_line) > 1:
            if macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]:
                signals['signals'].append({
                    'type': 'technical',
                    'signal': 'macd_bullish_cross',
                    'value': macd.iloc[-1] - signal_line.iloc[-1],
                    'description': 'MACD bullish crossover'
                })
                signals['strength'] += 1
            elif macd.iloc[-1] < signal_line.iloc[-1] and macd.iloc[-2] >= signal_line.iloc[-2]:
                signals['signals'].append({
                    'type': 'technical',
                    'signal': 'macd_bearish_cross',
                    'value': macd.iloc[-1] - signal_line.iloc[-1],
                    'description': 'MACD bearish crossover'
                })
                signals['strength'] -= 1
                
        return signals
        
    def detect_all_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect signals from all sources and combine"""
        self.logger.info("Detecting signals from all sources...")
        
        # Detect signals from each source
        price_signals = self.detect_price_signals(df)
        volume_signals = self.detect_volume_signals(df)
        weather_signals = self.detect_weather_signals(df)
        technical_signals = self.detect_technical_signals(df)
        
        # Combine all signals
        all_signals = {
            'timestamp': datetime.now(),
            'summary': {
                'total_signals': 0,
                'bullish_signals': 0,
                'bearish_signals': 0,
                'neutral_signals': 0,
                'composite_strength': 0,
                'signal_quality': 'low'
            },
            'price_signals': price_signals,
            'volume_signals': volume_signals,
            'weather_signals': weather_signals,
            'technical_signals': technical_signals,
            'recommendations': []
        }
        
        # Count signals by type
        total_strength = 0
        bullish_count = 0
        bearish_count = 0
        
        for source in [price_signals, volume_signals, weather_signals, technical_signals]:
            total_strength += source.get('strength', 0)
            
            for signal in source.get('signals', []):
                all_signals['summary']['total_signals'] += 1
                
                signal_type = signal.get('signal', '')
                if any(term in signal_type for term in ['bullish', 'surge', 'golden', 'oversold']):
                    bullish_count += 1
                elif any(term in signal_type for term in ['bearish', 'drought', 'death', 'overbought']):
                    bearish_count += 1
                    
        all_signals['summary']['bullish_signals'] = bullish_count
        all_signals['summary']['bearish_signals'] = bearish_count
        all_signals['summary']['neutral_signals'] = all_signals['summary']['total_signals'] - bullish_count - bearish_count
        all_signals['summary']['composite_strength'] = total_strength
        
        # Determine signal quality
        if abs(total_strength) >= 5:
            all_signals['summary']['signal_quality'] = 'high'
        elif abs(total_strength) >= 3:
            all_signals['summary']['signal_quality'] = 'medium'
        else:
            all_signals['summary']['signal_quality'] = 'low'
            
        # Generate recommendations
        if total_strength >= 5:
            all_signals['recommendations'].append("🚀 Strong BUY signals detected across multiple sources")
        elif total_strength >= 3:
            all_signals['recommendations'].append("📈 Moderate BUY signals - Consider entry")
        elif total_strength <= -5:
            all_signals['recommendations'].append("📉 Strong SELL signals detected across multiple sources")
        elif total_strength <= -3:
            all_signals['recommendations'].append("⚠️ Moderate SELL signals - Consider exit")
        else:
            all_signals['recommendations'].append("⏸️ Mixed signals - Wait for clarity")
            
        # Add specific signal recommendations
        if weather_signals['signals']:
            all_signals['recommendations'].append("🌦️ Weather patterns affecting production regions")
            
        if volume_signals['signals']:
            for sig in volume_signals['signals']:
                if 'surge' in sig.get('signal', ''):
                    all_signals['recommendations'].append("📊 Unusual trading volume detected")
                    
        return all_signals