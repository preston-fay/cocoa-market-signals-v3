"""
Unified Data Processing Pipeline

Combines all real data sources into a cohesive system for analysis and signal generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger
import json

from ..validation.data_validator import DataValidator, DataSource, DataPoint


class UnifiedDataPipeline:
    """
    Processes and combines all data sources with validation
    """
    
    def __init__(self):
        self.data_dir = Path("data")
        self.validator = DataValidator()
        logger.add("logs/unified_pipeline.log", rotation="1 day")
        
        # Data paths
        self.price_data_path = self.data_dir / "historical/prices/cocoa_daily_prices_2yr.csv"
        self.weather_data_path = self.data_dir / "historical/weather/all_locations_weather_2yr.csv"
        self.economics_data_path = self.data_dir / "historical/economics"
        
    def load_price_data(self) -> pd.DataFrame:
        """Load and validate daily price data"""
        logger.info("Loading daily price data")
        
        if not self.price_data_path.exists():
            logger.error(f"Price data not found at {self.price_data_path}")
            return pd.DataFrame()
            
        # Load data
        df = pd.read_csv(self.price_data_path, index_col=0, parse_dates=True)
        
        # Validate each price point
        validated_count = 0
        for idx, row in df.iterrows():
            source = DataSource(
                source_name="Yahoo Finance",
                source_type="api",
                source_url="https://finance.yahoo.com",
                retrieval_time=datetime.now(),
                data_hash=self.validator.create_source_hash(row.to_dict()),
                verified=True
            )
            
            # Validate closing price
            # Convert timestamp to timezone-naive for validation
            timestamp = idx.tz_localize(None) if idx.tz else idx
            
            data_point = DataPoint(
                timestamp=timestamp,
                value=row['cocoa_cc_close'],
                metric_name="price_usd_per_ton",
                source=source,
                confidence_level=0.95,
                validation_status="verified"
            )
            
            result = self.validator.validate_data_point(data_point)
            if result.is_valid:
                validated_count += 1
                
        logger.info(f"Validated {validated_count}/{len(df)} price points")
        return df
    
    def load_weather_data(self) -> pd.DataFrame:
        """Load and validate weather data"""
        logger.info("Loading weather data")
        
        if not self.weather_data_path.exists():
            logger.error(f"Weather data not found at {self.weather_data_path}")
            return pd.DataFrame()
            
        # Load data
        df = pd.read_csv(self.weather_data_path, index_col=0, parse_dates=True)
        
        # Pivot by location for easier analysis
        weather_pivoted = {}
        for location in df['location'].unique():
            location_data = df[df['location'] == location].copy()
            location_data = location_data.drop(columns=['location', 'country'])
            weather_pivoted[location] = location_data
            
        return weather_pivoted
    
    def load_economics_data(self) -> Dict[str, pd.DataFrame]:
        """Load economic indicators"""
        logger.info("Loading economics data")
        
        economics_data = {}
        
        # Load inflation data
        inflation_path = self.economics_data_path / "inflation_currency_data.json"
        if inflation_path.exists():
            with open(inflation_path, 'r') as f:
                data = json.load(f)
                
            # Convert to DataFrame
            inflation_records = []
            for date, countries in data['inflation'].items():
                for country, metrics in countries.items():
                    record = {
                        'date': pd.to_datetime(date + '-01'),
                        'country': country,
                        'monthly_inflation': metrics['monthly'],
                        'yoy_inflation': metrics['yoy'],
                        'food_inflation': metrics.get('food_inflation', np.nan)
                    }
                    inflation_records.append(record)
                    
            economics_data['inflation'] = pd.DataFrame(inflation_records)
            
            # Convert currency data if available
            if 'currency' in data:
                currency_records = []
                for date, pairs in data['currency'].items():
                    for pair, rate in pairs.items():
                        record = {
                            'date': pd.to_datetime(date + '-01'),
                            'pair': pair,
                            'rate': rate
                        }
                        currency_records.append(record)
                        
                economics_data['currency'] = pd.DataFrame(currency_records)
            elif 'currency_impact' in data:
                # Handle currency impact data
                economics_data['currency_impact'] = data['currency_impact']
        
        # Load shipping data
        shipping_path = self.economics_data_path / "shipping_costs.json"
        if shipping_path.exists():
            with open(shipping_path, 'r') as f:
                shipping_data = json.load(f)
                
            shipping_records = []
            for date, data in shipping_data.items():
                # Extract average container rates
                if 'container_rates' in data:
                    container_avg = []
                    for route, rates in data['container_rates'].items():
                        if isinstance(rates, dict) and '20ft' in rates:
                            container_avg.append(rates['20ft'])
                    
                    if container_avg:
                        record = {
                            'date': pd.to_datetime(date + '-01'),
                            'container_rate_avg': np.mean(container_avg),
                            'bulk_rate_avg': data.get('bulk_rates', {}).get('avg_rate', np.nan)
                        }
                        shipping_records.append(record)
                    
            if shipping_records:
                economics_data['shipping'] = pd.DataFrame(shipping_records)
            
        return economics_data
    
    def create_feature_matrix(self, start_date: Optional[str] = None, 
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Create unified feature matrix for modeling
        """
        logger.info("Creating unified feature matrix")
        
        # Load all data
        price_df = self.load_price_data()
        weather_data = self.load_weather_data()
        economics_data = self.load_economics_data()
        
        if price_df.empty:
            logger.error("No price data available")
            return pd.DataFrame()
            
        # Start with price data as base
        feature_df = price_df[['cocoa_cc_close', 'cocoa_cc_volume', 
                              'cocoa_cc_volatility_30d', 'cocoa_cc_rsi']].copy()
        
        # Rename for clarity
        feature_df.columns = ['price', 'volume', 'volatility_30d', 'rsi']
        
        # Add price features
        feature_df['returns_1d'] = feature_df['price'].pct_change()
        feature_df['returns_7d'] = feature_df['price'].pct_change(7)
        feature_df['returns_30d'] = feature_df['price'].pct_change(30)
        
        # Add weather features (average across locations)
        if weather_data:
            weather_features = []
            for location, df in weather_data.items():
                # Ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                # Resample to daily if needed and forward fill
                df_daily = df.resample('D').mean().ffill()
                
                # Add location prefix
                df_daily.columns = [f"{location}_{col}" for col in df_daily.columns]
                weather_features.append(df_daily)
                
            # Combine all weather data
            all_weather = pd.concat(weather_features, axis=1)
            
            # Calculate regional averages - create a temporary aligned df
            temp_df = pd.DataFrame(index=feature_df.index)
            temp_df = temp_df.join(all_weather, how='left')
            
            feature_df['avg_temp'] = temp_df[[col for col in temp_df.columns 
                                            if 'temp_mean' in col]].mean(axis=1)
            feature_df['avg_rainfall'] = temp_df[[col for col in temp_df.columns 
                                                if 'precipitation' in col]].mean(axis=1)
            feature_df['avg_weather_risk'] = temp_df[[col for col in temp_df.columns 
                                                    if 'weather_risk' in col]].mean(axis=1)
            
            # Merge all weather data
            for col in all_weather.columns:
                feature_df[col] = temp_df[col]
        
        # Add economic features
        if 'inflation' in economics_data:
            # Average inflation across countries
            inflation_avg = economics_data['inflation'].groupby('date')[
                ['yoy_inflation', 'food_inflation']
            ].mean()
            
            # Resample to daily and forward fill
            inflation_daily = inflation_avg.resample('D').ffill()
            feature_df = feature_df.join(inflation_daily, how='left')
        
        if 'currency' in economics_data:
            # Pivot currency data
            currency_pivot = economics_data['currency'].pivot(
                index='date', columns='pair', values='rate'
            )
            currency_daily = currency_pivot.resample('D').ffill()
            feature_df = feature_df.join(currency_daily, how='left')
        
        if 'shipping' in economics_data:
            # Set shipping costs with date index
            shipping_df = economics_data['shipping'].set_index('date')
            shipping_daily = shipping_df.resample('D').ffill()
            feature_df = feature_df.join(shipping_daily, how='left')
        
        # Add time features
        # Ensure we have a proper datetime index
        if not isinstance(feature_df.index, pd.DatetimeIndex):
            feature_df.index = pd.to_datetime(feature_df.index)
            
        # Remove timezone if present to avoid issues
        if hasattr(feature_df.index, 'tz') and feature_df.index.tz is not None:
            feature_df.index = feature_df.index.tz_localize(None)
        
        feature_df['day_of_week'] = feature_df.index.dayofweek
        feature_df['month'] = feature_df.index.month
        feature_df['quarter'] = feature_df.index.quarter
        feature_df['is_harvest_season'] = feature_df['month'].isin([10, 11, 12, 1, 2])
        
        # Apply date filter if specified
        if start_date:
            feature_df = feature_df[feature_df.index >= start_date]
        if end_date:
            feature_df = feature_df[feature_df.index <= end_date]
            
        # Drop rows with NaN in critical features
        critical_features = ['price', 'volume', 'volatility_30d']
        feature_df = feature_df.dropna(subset=critical_features)
        
        # Forward fill remaining NaNs
        feature_df = feature_df.ffill()
        
        logger.info(f"Created feature matrix with shape: {feature_df.shape}")
        logger.info(f"Features: {list(feature_df.columns)}")
        
        return feature_df
    
    def generate_signals(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on multiple indicators
        """
        logger.info("Generating trading signals")
        
        signals = pd.DataFrame(index=feature_df.index)
        
        # Price momentum signals
        signals['momentum_signal'] = np.where(
            (feature_df['returns_7d'] > 0.05) & (feature_df['rsi'] < 70), 1,
            np.where((feature_df['returns_7d'] < -0.05) & (feature_df['rsi'] > 30), -1, 0)
        )
        
        # Volatility signals
        vol_mean = feature_df['volatility_30d'].rolling(90).mean()
        signals['volatility_signal'] = np.where(
            feature_df['volatility_30d'] > vol_mean * 1.5, -1,  # High vol = caution
            np.where(feature_df['volatility_30d'] < vol_mean * 0.7, 1, 0)  # Low vol = opportunity
        )
        
        # Weather risk signals
        if 'avg_weather_risk' in feature_df.columns:
            signals['weather_signal'] = np.where(
                feature_df['avg_weather_risk'] > 0.3, -1,  # High risk = bearish
                np.where(feature_df['avg_weather_risk'] < 0.1, 1, 0)  # Low risk = bullish
            )
        else:
            signals['weather_signal'] = 0
            
        # Seasonal signals
        signals['seasonal_signal'] = np.where(
            feature_df['is_harvest_season'] & (feature_df['month'].isin([10, 11])), -1,  # Early harvest
            np.where(feature_df['is_harvest_season'] & (feature_df['month'].isin([1, 2])), 1, 0)  # Late harvest
        )
        
        # Composite signal
        signal_cols = ['momentum_signal', 'volatility_signal', 'weather_signal', 'seasonal_signal']
        signals['composite_signal'] = signals[signal_cols].mean(axis=1)
        
        # Final signal with thresholds
        signals['final_signal'] = np.where(
            signals['composite_signal'] >= 0.5, 1,  # Strong buy
            np.where(signals['composite_signal'] <= -0.5, -1, 0)  # Strong sell
        )
        
        # Signal strength
        signals['signal_strength'] = np.abs(signals['composite_signal'])
        
        # Add features for context
        signals['price'] = feature_df['price']
        signals['volume'] = feature_df['volume']
        
        return signals
    
    def save_processed_data(self, feature_df: pd.DataFrame, 
                          signals_df: pd.DataFrame) -> None:
        """Save processed data for dashboard and analysis"""
        
        # Save feature matrix
        feature_path = self.data_dir / "processed/feature_matrix.csv"
        feature_path.parent.mkdir(exist_ok=True)
        feature_df.to_csv(feature_path)
        logger.info(f"Saved feature matrix to {feature_path}")
        
        # Save signals
        signals_path = self.data_dir / "processed/signals.csv"
        signals_df.to_csv(signals_path)
        logger.info(f"Saved signals to {signals_path}")
        
        # Create summary for dashboard
        summary = {
            "last_updated": datetime.now().isoformat(),
            "data_period": {
                "start": feature_df.index.min().strftime("%Y-%m-%d"),
                "end": feature_df.index.max().strftime("%Y-%m-%d")
            },
            "current_signal": {
                "signal": int(signals_df['final_signal'].iloc[-1]),
                "strength": float(signals_df['signal_strength'].iloc[-1]),
                "price": float(signals_df['price'].iloc[-1]),
                "date": signals_df.index[-1].strftime("%Y-%m-%d")
            },
            "feature_count": len(feature_df.columns),
            "signal_distribution": {
                "buy": int((signals_df['final_signal'] == 1).sum()),
                "sell": int((signals_df['final_signal'] == -1).sum()),
                "neutral": int((signals_df['final_signal'] == 0).sum())
            }
        }
        
        summary_path = self.data_dir / "processed/pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Saved pipeline summary to {summary_path}")


if __name__ == "__main__":
    # Initialize pipeline
    pipeline = UnifiedDataPipeline()
    
    # Create feature matrix
    features = pipeline.create_feature_matrix()
    
    if not features.empty:
        # Generate signals
        signals = pipeline.generate_signals(features)
        
        # Save processed data
        pipeline.save_processed_data(features, signals)
        
        print(f"\nPipeline completed successfully!")
        print(f"Features shape: {features.shape}")
        print(f"Current signal: {signals['final_signal'].iloc[-1]}")
        print(f"Signal strength: {signals['signal_strength'].iloc[-1]:.2f}")
    else:
        print("Failed to create feature matrix")