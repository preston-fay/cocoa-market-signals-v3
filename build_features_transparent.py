#!/usr/bin/env python3
"""
BUILD FEATURES WITH FULL TRANSPARENCY
Every step is logged and verified
"""
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta

class TransparentFeatureBuilder:
    def __init__(self):
        self.conn = sqlite3.connect("data/cocoa_market_signals_real.db")
        self.features_log = []
        
    def log(self, message, data=None):
        """Log every action for transparency"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        if data is not None:
            print(f"    Data shape: {data.shape if hasattr(data, 'shape') else len(data)}")
            if hasattr(data, 'head'):
                print(f"    Sample:\n{data.head(3)}")
        self.features_log.append({"time": datetime.now(), "message": message})
        
    def load_price_data(self):
        """Load price data and show exactly what we have"""
        self.log("Loading price data from database...")
        
        query = """
        SELECT date, open, high, low, close, volume 
        FROM price_data 
        ORDER BY date
        """
        prices_df = pd.read_sql_query(query, self.conn, parse_dates=['date'])
        prices_df.set_index('date', inplace=True)
        
        self.log(f"Loaded {len(prices_df)} price records", prices_df[['close', 'volume']])
        self.log(f"Date range: {prices_df.index.min()} to {prices_df.index.max()}")
        
        # Calculate technical features
        self.log("Calculating technical indicators...")
        
        # Returns
        prices_df['return_1d'] = prices_df['close'].pct_change(1)
        prices_df['return_5d'] = prices_df['close'].pct_change(5)
        prices_df['return_20d'] = prices_df['close'].pct_change(20)
        
        # Moving averages
        prices_df['sma_10'] = prices_df['close'].rolling(10).mean()
        prices_df['sma_30'] = prices_df['close'].rolling(30).mean()
        prices_df['sma_ratio'] = prices_df['sma_10'] / prices_df['sma_30']
        
        # Volatility
        prices_df['volatility_20d'] = prices_df['return_1d'].rolling(20).std()
        
        # Volume indicators
        prices_df['volume_sma_20'] = prices_df['volume'].rolling(20).mean()
        prices_df['volume_ratio'] = prices_df['volume'] / prices_df['volume_sma_20']
        
        self.log("Technical features created", prices_df[['return_1d', 'sma_ratio', 'volatility_20d']].tail(3))
        
        return prices_df
        
    def load_weather_data(self):
        """Load weather and aggregate by date"""
        self.log("Loading weather data...")
        
        query = """
        SELECT date, location, temp_mean, temp_max, temp_min, rainfall, humidity
        FROM weather_data
        WHERE date >= '2023-07-01'
        ORDER BY date, location
        """
        weather_df = pd.read_sql_query(query, self.conn, parse_dates=['date'])
        
        self.log(f"Loaded {len(weather_df)} weather records from {weather_df['location'].nunique()} locations")
        
        # Aggregate by date
        self.log("Aggregating weather by date...")
        
        weather_agg = weather_df.groupby('date').agg({
            'temp_mean': ['mean', 'std', 'min', 'max'],
            'rainfall': ['sum', 'max', 'mean'],
            'humidity': ['mean', 'std']
        })
        
        # Flatten column names
        weather_agg.columns = ['_'.join(col).strip() for col in weather_agg.columns]
        
        # Add weather anomalies
        self.log("Calculating weather anomalies...")
        
        # Temperature anomaly (deviation from 30-day average)
        weather_agg['temp_anomaly'] = weather_agg['temp_mean_mean'] - weather_agg['temp_mean_mean'].rolling(30).mean()
        
        # Extreme rainfall indicator
        weather_agg['extreme_rain'] = (weather_agg['rainfall_max'] > weather_agg['rainfall_max'].quantile(0.9)).astype(int)
        
        self.log("Weather features created", weather_agg[['temp_mean_mean', 'rainfall_sum', 'temp_anomaly']].tail(3))
        
        return weather_agg
        
    def load_news_sentiment(self):
        """Load news sentiment aggregated by date"""
        self.log("Loading news sentiment data...")
        
        query = """
        SELECT 
            DATE(published_date) as date,
            sentiment_score,
            source
        FROM news_articles
        WHERE sentiment_score IS NOT NULL
        AND published_date >= '2023-07-01'
        """
        news_df = pd.read_sql_query(query, self.conn, parse_dates=['date'])
        
        self.log(f"Loaded {len(news_df)} news articles with sentiment")
        
        # Show sentiment distribution
        self.log("Sentiment distribution:")
        print(f"    Positive (>0.1): {(news_df['sentiment_score'] > 0.1).sum()}")
        print(f"    Negative (<-0.1): {(news_df['sentiment_score'] < -0.1).sum()}")
        print(f"    Neutral: {((news_df['sentiment_score'] >= -0.1) & (news_df['sentiment_score'] <= 0.1)).sum()}")
        
        # Aggregate by date
        self.log("Aggregating sentiment by date...")
        
        sentiment_daily = news_df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'min', 'max', 'count']
        })
        
        sentiment_daily.columns = ['sentiment_mean', 'sentiment_std', 'sentiment_min', 'sentiment_max', 'article_count']
        
        # Add rolling sentiment features
        sentiment_daily['sentiment_ma_7d'] = sentiment_daily['sentiment_mean'].rolling(7).mean()
        sentiment_daily['sentiment_momentum'] = sentiment_daily['sentiment_mean'] - sentiment_daily['sentiment_ma_7d']
        
        self.log("Sentiment features created", sentiment_daily[['sentiment_mean', 'article_count', 'sentiment_momentum']].tail(3))
        
        return sentiment_daily
        
    def load_trade_data(self):
        """Load Comtrade export data"""
        self.log("Loading Comtrade export data...")
        
        query = """
        SELECT 
            year || '-' || printf('%02d', month) || '-15' as date,
            reporter_country,
            SUM(quantity_kg) as total_quantity,
            SUM(trade_value_usd) as total_value,
            AVG(unit_price) as avg_price
        FROM comtrade_data
        WHERE partner_country = 'World'
        AND reporter_country IN ('Côte d''Ivoire', 'Ghana')
        GROUP BY year, month, reporter_country
        """
        trade_df = pd.read_sql_query(query, self.conn, parse_dates=['date'])
        
        self.log(f"Loaded {len(trade_df)} monthly trade records")
        
        # Pivot by country
        trade_pivot = trade_df.pivot_table(
            index='date',
            columns='reporter_country',
            values=['total_quantity', 'total_value'],
            aggfunc='sum'
        )
        
        # Flatten columns
        trade_pivot.columns = ['_'.join(col).strip() for col in trade_pivot.columns]
        
        # Calculate market concentration
        self.log("Calculating trade features...")
        
        # Total exports
        trade_pivot['total_exports_kg'] = trade_pivot[[col for col in trade_pivot.columns if 'quantity' in col]].sum(axis=1)
        
        # Ivory Coast market share
        if 'total_quantity_Côte d\'Ivoire' in trade_pivot.columns:
            trade_pivot['ic_market_share'] = trade_pivot['total_quantity_Côte d\'Ivoire'] / trade_pivot['total_exports_kg']
        
        self.log("Trade features created", trade_pivot[['total_exports_kg']].tail(3))
        
        return trade_pivot
        
    def merge_all_features(self, prices_df, weather_agg, sentiment_daily, trade_pivot):
        """Merge all features by date - THE CRITICAL STEP"""
        self.log("\n" + "="*60)
        self.log("MERGING ALL FEATURES BY DATE - THIS IS THE CRITICAL STEP")
        self.log("="*60)
        
        # Start with prices as base
        features_df = prices_df.copy()
        self.log(f"Starting with {len(features_df)} price records")
        
        # Merge weather
        self.log("Merging weather data...")
        features_df = features_df.join(weather_agg, how='left')
        self.log(f"After weather merge: {len(features_df)} records, {features_df.shape[1]} features")
        
        # Check weather coverage
        weather_coverage = features_df['temp_mean_mean'].notna().sum() / len(features_df)
        self.log(f"Weather data coverage: {weather_coverage:.1%}")
        
        # Merge sentiment
        self.log("Merging sentiment data...")
        features_df = features_df.join(sentiment_daily, how='left')
        self.log(f"After sentiment merge: {len(features_df)} records, {features_df.shape[1]} features")
        
        # Check sentiment coverage
        sentiment_coverage = features_df['sentiment_mean'].notna().sum() / len(features_df)
        self.log(f"Sentiment data coverage: {sentiment_coverage:.1%}")
        
        # Merge trade data (monthly, so we'll forward fill)
        self.log("Merging trade data...")
        features_df = features_df.join(trade_pivot, how='left')
        
        # Forward fill trade data (monthly data applied to daily)
        trade_cols = [col for col in trade_pivot.columns]
        features_df[trade_cols] = features_df[trade_cols].fillna(method='ffill')
        
        self.log(f"After trade merge: {len(features_df)} records, {features_df.shape[1]} features")
        
        # Fill missing values appropriately
        self.log("Handling missing values...")
        
        # Sentiment: 0 means no news that day
        features_df['sentiment_mean'].fillna(0, inplace=True)
        features_df['sentiment_std'].fillna(0, inplace=True)
        features_df['article_count'].fillna(0, inplace=True)
        
        # Weather: forward fill (weather changes slowly)
        weather_cols = [col for col in features_df.columns if 'temp' in col or 'rainfall' in col or 'humidity' in col]
        features_df[weather_cols] = features_df[weather_cols].fillna(method='ffill', limit=3)
        
        # Show final coverage
        self.log("\nFINAL FEATURE COVERAGE:")
        for col in ['close', 'temp_mean_mean', 'sentiment_mean', 'total_exports_kg']:
            if col in features_df.columns:
                coverage = features_df[col].notna().sum() / len(features_df)
                print(f"    {col}: {coverage:.1%}")
                
        return features_df
        
    def create_targets(self, features_df):
        """Create prediction targets"""
        self.log("\nCreating prediction targets...")
        
        # Future returns
        features_df['return_1d_future'] = features_df['return_1d'].shift(-1)
        features_df['return_5d_future'] = features_df['return_5d'].shift(-5)
        features_df['return_20d_future'] = features_df['return_20d'].shift(-20)
        
        # Direction (up/down)
        features_df['direction_1d'] = (features_df['return_1d_future'] > 0).astype(int)
        features_df['direction_5d'] = (features_df['return_5d_future'] > 0).astype(int)
        features_df['direction_20d'] = (features_df['return_20d_future'] > 0).astype(int)
        
        self.log("Targets created")
        
        return features_df
        
    def save_features(self, features_df):
        """Save the final dataset"""
        self.log("\nSaving final dataset...")
        
        # Drop rows with NaN in critical columns
        before_drop = len(features_df)
        features_df = features_df.dropna(subset=['close', 'return_1d', 'direction_5d'])
        after_drop = len(features_df)
        
        self.log(f"Dropped {before_drop - after_drop} rows with missing critical values")
        self.log(f"Final dataset: {len(features_df)} samples with {features_df.shape[1]} features")
        
        # Save
        features_df.to_csv('data/processed/features_with_real_dates.csv')
        
        # Show final summary
        self.log("\nFINAL DATASET SUMMARY:")
        print(f"    Date range: {features_df.index.min()} to {features_df.index.max()}")
        print(f"    Total samples: {len(features_df)}")
        print(f"    Total features: {features_df.shape[1]}")
        print(f"    Features with >90% coverage: {(features_df.notna().sum() / len(features_df) > 0.9).sum()}")
        
        # Show feature groups
        price_features = [col for col in features_df.columns if any(x in col for x in ['return', 'sma', 'volume', 'close', 'volatility'])]
        weather_features = [col for col in features_df.columns if any(x in col for x in ['temp', 'rain', 'humidity', 'anomaly'])]
        sentiment_features = [col for col in features_df.columns if any(x in col for x in ['sentiment', 'article'])]
        trade_features = [col for col in features_df.columns if any(x in col for x in ['export', 'trade', 'market_share'])]
        
        print(f"\n    Feature breakdown:")
        print(f"    - Price/Technical: {len(price_features)} features")
        print(f"    - Weather: {len(weather_features)} features")
        print(f"    - Sentiment/News: {len(sentiment_features)} features")
        print(f"    - Trade/Export: {len(trade_features)} features")
        
        return features_df
        
    def run(self):
        """Execute the full pipeline with transparency"""
        print("="*60)
        print("TRANSPARENT FEATURE BUILDING PIPELINE")
        print("="*60)
        
        # Load each data source
        prices_df = self.load_price_data()
        weather_agg = self.load_weather_data()
        sentiment_daily = self.load_news_sentiment()
        trade_pivot = self.load_trade_data()
        
        # Merge everything
        features_df = self.merge_all_features(prices_df, weather_agg, sentiment_daily, trade_pivot)
        
        # Create targets
        features_df = self.create_targets(features_df)
        
        # Save
        final_df = self.save_features(features_df)
        
        self.conn.close()
        
        print("\n✅ Feature building complete!")
        print(f"📄 Output saved to: data/processed/features_with_real_dates.csv")
        
        return final_df

if __name__ == "__main__":
    builder = TransparentFeatureBuilder()
    features_df = builder.run()