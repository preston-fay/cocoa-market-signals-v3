#!/usr/bin/env python3
"""
Comprehensive Feature Extraction Pipeline
Integrates all data sources into a unified feature matrix
100% REAL data - NO synthetic features
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import datetime as dt
from typing import Dict, List, Tuple, Optional
from sqlmodel import Session, select
# import talib  # Optional - using manual calculations instead
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.database import engine
from app.models.price_data import PriceData
from app.models.weather_data import WeatherData
from app.models.news_article import NewsArticle
from app.models.trade_data import TradeData
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveFeatureExtractor:
    """
    Extracts features from all data sources with proper temporal alignment
    """
    
    def __init__(self):
        self.feature_groups = {
            'price': [],
            'weather': [],
            'sentiment': [],
            'trade': [],
            'interaction': []
        }
    
    def _to_datetime(self, date_obj):
        """Convert date or datetime to datetime object"""
        if isinstance(date_obj, datetime):
            return date_obj
        elif isinstance(date_obj, dt.date):
            return datetime.combine(date_obj, datetime.min.time())
        else:
            return date_obj
        
    def extract_all_features(self, target_date: datetime, 
                           lookback_days: int = 90) -> pd.DataFrame:
        """
        Extract all features for a given target date
        Only uses data available before target_date
        """
        logger.info(f"Extracting features for {target_date.date()}")
        
        # Get price features
        price_features = self._extract_price_features(target_date, lookback_days)
        
        # Get weather features
        weather_features = self._extract_weather_features(target_date, lookback_days)
        
        # Get sentiment features
        sentiment_features = self._extract_sentiment_features(target_date, lookback_days)
        
        # Get trade features
        trade_features = self._extract_trade_features(target_date, lookback_days)
        
        # Get interaction features
        interaction_features = self._extract_interaction_features(
            price_features, weather_features, sentiment_features, trade_features
        )
        
        # Combine all features
        all_features = pd.concat([
            price_features,
            weather_features,
            sentiment_features,
            trade_features,
            interaction_features
        ], axis=1)
        
        # Handle missing values
        all_features = self._handle_missing_values(all_features)
        
        # Scale features
        all_features = self._scale_features(all_features)
        
        logger.info(f"Extracted {len(all_features.columns)} features")
        
        return all_features
    
    def _extract_price_features(self, target_date: datetime, 
                               lookback_days: int) -> pd.DataFrame:
        """Extract technical and statistical features from price data"""
        
        with Session(engine) as session:
            # Get historical prices
            start_date = (target_date - timedelta(days=lookback_days + 30)).date()  # Extra for indicators
            target_date_only = target_date.date()
            
            prices = session.exec(
                select(PriceData)
                .where(PriceData.date >= start_date)
                .where(PriceData.date < target_date_only)
                .where(PriceData.source == "Yahoo Finance")
                .order_by(PriceData.date)
            ).all()
            
            if not prices:
                logger.warning("No price data found")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'date': p.date,
                'close': p.price,
                'high': p.high or p.price,
                'low': p.low or p.price,
                'volume': p.volume or 0
            } for p in prices])
            
            df.set_index('date', inplace=True)
            
            features = {}
            
            # Returns
            features['return_1d'] = df['close'].pct_change(1).iloc[-1]
            features['return_7d'] = df['close'].pct_change(7).iloc[-1]
            features['return_30d'] = df['close'].pct_change(30).iloc[-1]
            
            # Volatility
            returns = df['close'].pct_change()
            features['volatility_7d'] = returns.rolling(7).std().iloc[-1]
            features['volatility_30d'] = returns.rolling(30).std().iloc[-1]
            
            # Technical indicators (manual calculations)
            # RSI
            features['rsi_14'] = self._calculate_rsi(df['close'], 14)
            
            # MACD
            macd_line, signal_line, histogram = self._calculate_macd(df['close'])
            features['macd'] = macd_line
            features['macd_signal'] = signal_line
            features['macd_histogram'] = histogram
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'], 20)
            features['bb_position'] = (df['close'].iloc[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            
            # Moving averages
            features['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
            features['sma_50'] = df['close'].rolling(50).mean().iloc[-1]
            features['price_to_sma20'] = df['close'].iloc[-1] / features['sma_20'] - 1
            features['price_to_sma50'] = df['close'].iloc[-1] / features['sma_50'] - 1
            
            # Momentum
            features['momentum_10d'] = df['close'].iloc[-1] - df['close'].iloc[-11] if len(df) > 10 else 0
            
            # Volume features
            if df['volume'].sum() > 0:
                features['volume_ratio'] = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                features['volume_trend'] = df['volume'].rolling(7).mean().iloc[-1] / df['volume'].rolling(30).mean().iloc[-1]
            else:
                features['volume_ratio'] = 1.0
                features['volume_trend'] = 1.0
            
            return pd.DataFrame([features])
    
    def _extract_weather_features(self, target_date: datetime, 
                                 lookback_days: int) -> pd.DataFrame:
        """Extract weather risk features aggregated by region"""
        
        with Session(engine) as session:
            # Get weather data
            start_date = (target_date - timedelta(days=lookback_days)).date()
            target_date_only = target_date.date()
            
            weather = session.exec(
                select(WeatherData)
                .where(WeatherData.date >= start_date)
                .where(WeatherData.date < target_date_only)
                .order_by(WeatherData.date)
            ).all()
            
            if not weather:
                logger.warning("No weather data found")
                return pd.DataFrame()
            
            # Group by country
            regions = {}
            for w in weather:
                if w.country not in regions:
                    regions[w.country] = []
                regions[w.country].append(w)
            
            features = {}
            
            # Aggregate features by region
            for region, data in regions.items():
                # Recent averages
                recent_cutoff = target_date - timedelta(days=7)
                recent_data = [d for d in data if datetime.combine(d.date, datetime.min.time()) >= recent_cutoff]
                if recent_data:
                    drought_risks = [d.drought_risk for d in recent_data if d.drought_risk is not None]
                    flood_risks = [d.flood_risk for d in recent_data if d.flood_risk is not None]
                    disease_risks = [d.disease_risk for d in recent_data if d.disease_risk is not None]
                    
                    features[f'{region}_drought_risk_7d'] = np.mean(drought_risks) if drought_risks else 0
                    features[f'{region}_flood_risk_7d'] = np.mean(flood_risks) if flood_risks else 0
                    features[f'{region}_disease_risk_7d'] = np.mean(disease_risks) if disease_risks else 0
                
                # Monthly averages
                monthly_cutoff = target_date - timedelta(days=30)
                monthly_data = [d for d in data if datetime.combine(d.date, datetime.min.time()) >= monthly_cutoff]
                if monthly_data:
                    features[f'{region}_temp_anomaly_30d'] = np.mean([
                        d.temp_max - 28 for d in monthly_data  # 28°C is optimal
                    ])
                    features[f'{region}_rainfall_total_30d'] = sum([d.precipitation_mm for d in monthly_data])
            
            # Overall risk scores
            recent_cutoff = target_date - timedelta(days=7)
            all_recent = [d for d in weather if datetime.combine(d.date, datetime.min.time()) >= recent_cutoff]
            if all_recent:
                drought_risks = [d.drought_risk for d in all_recent if d.drought_risk is not None]
                flood_risks = [d.flood_risk for d in all_recent if d.flood_risk is not None]
                disease_risks = [d.disease_risk for d in all_recent if d.disease_risk is not None]
                
                features['overall_drought_risk'] = np.mean(drought_risks) if drought_risks else 0
                features['overall_flood_risk'] = np.mean(flood_risks) if flood_risks else 0
                features['overall_disease_risk'] = np.mean(disease_risks) if disease_risks else 0
                features['max_regional_risk'] = max(
                    features.get('overall_drought_risk', 0),
                    features.get('overall_flood_risk', 0),
                    features.get('overall_disease_risk', 0)
                )
            
            # Extreme weather events
            monthly_cutoff_dt = target_date - timedelta(days=30)
            features['extreme_heat_days_30d'] = sum([
                1 for d in weather 
                if datetime.combine(d.date, datetime.min.time()) >= monthly_cutoff_dt and d.temp_max > 35
            ])
            features['heavy_rain_days_30d'] = sum([
                1 for d in weather 
                if datetime.combine(d.date, datetime.min.time()) >= monthly_cutoff_dt and d.precipitation_mm > 50
            ])
            
            return pd.DataFrame([features])
    
    def _extract_sentiment_features(self, target_date: datetime, 
                                   lookback_days: int) -> pd.DataFrame:
        """Extract news sentiment features"""
        
        with Session(engine) as session:
            # Get analyzed articles
            start_date = target_date - timedelta(days=lookback_days)
            
            articles = session.exec(
                select(NewsArticle)
                .where(NewsArticle.published_date >= start_date)
                .where(NewsArticle.published_date < target_date)
                .where(NewsArticle.sentiment_score.is_not(None))
                .order_by(NewsArticle.published_date)
            ).all()
            
            if not articles:
                logger.warning("No sentiment data found")
                return pd.DataFrame()
            
            features = {}
            
            # Overall sentiment
            sentiments = [a.sentiment_score for a in articles]
            features['sentiment_mean'] = np.mean(sentiments)
            features['sentiment_std'] = np.std(sentiments)
            features['sentiment_skew'] = self._calculate_skewness(sentiments)
            
            # Recent sentiment (7 days)
            recent_articles = [a for a in articles if a.published_date >= target_date - timedelta(days=7)]
            if recent_articles:
                recent_sentiments = [a.sentiment_score for a in recent_articles]
                features['sentiment_mean_7d'] = np.mean(recent_sentiments)
                features['sentiment_momentum'] = features['sentiment_mean_7d'] - features['sentiment_mean']
                features['article_count_7d'] = len(recent_articles)
            
            # Sentiment by label
            labels = [a.sentiment_label for a in articles if a.sentiment_label]
            if labels:
                label_counts = pd.Series(labels).value_counts()
                total = len(labels)
                features['positive_ratio'] = label_counts.get('positive', 0) / total
                features['negative_ratio'] = label_counts.get('negative', 0) / total
                features['very_negative_ratio'] = label_counts.get('very_negative', 0) / total
            
            # Topic analysis
            all_topics = []
            for a in articles:
                if a.topics:
                    all_topics.extend(a.topics.split(','))
            
            if all_topics:
                topic_counts = pd.Series(all_topics).value_counts()
                features['weather_topic_ratio'] = topic_counts.get('weather', 0) / len(articles)
                features['disease_topic_ratio'] = topic_counts.get('disease', 0) / len(articles)
                features['market_topic_ratio'] = topic_counts.get('market', 0) / len(articles)
                features['policy_topic_ratio'] = topic_counts.get('policy', 0) / len(articles)
            
            # High impact articles
            features['high_impact_count'] = sum([
                1 for a in articles if abs(a.sentiment_score) > 0.7
            ])
            
            return pd.DataFrame([features])
    
    def _extract_trade_features(self, target_date: datetime, 
                               lookback_days: int) -> pd.DataFrame:
        """Extract trade volume and pattern features"""
        
        with Session(engine) as session:
            # Get trade data (with lag for reporting delays)
            trade_lag = 60  # Assume 60-day reporting lag
            start_date = target_date - timedelta(days=lookback_days + trade_lag)
            end_date = target_date - timedelta(days=trade_lag)
            
            trades = session.exec(
                select(TradeData)
                .where(TradeData.period >= start_date)
                .where(TradeData.period < end_date)
                .order_by(TradeData.period)
            ).all()
            
            if not trades:
                logger.warning("No trade data found")
                return pd.DataFrame()
            
            features = {}
            
            # Group by exporter
            exporters = {}
            for t in trades:
                if t.reporter_country not in exporters:
                    exporters[t.reporter_country] = []
                exporters[t.reporter_country].append(t)
            
            # Total export volumes (convert tons to kg)
            total_volume = sum([t.quantity_tons * 1000 for t in trades if t.quantity_tons])
            total_value = sum([t.trade_value_usd for t in trades if t.trade_value_usd])
            
            features['total_export_volume'] = total_volume
            features['total_export_value'] = total_value
            features['avg_price_per_kg'] = total_value / total_volume if total_volume > 0 else 0
            
            # By major exporters
            for country in ['Ghana', 'Côte d\'Ivoire', 'Nigeria', 'Cameroon']:
                if country in exporters:
                    country_data = exporters[country]
                    country_volume = sum([t.quantity_tons * 1000 for t in country_data if t.quantity_tons])
                    features[f'{country}_export_share'] = country_volume / total_volume if total_volume > 0 else 0
                    
                    # Year-over-year change  
                    recent = [t for t in country_data if self._to_datetime(t.period) >= end_date - timedelta(days=90)]
                    old = [t for t in country_data if self._to_datetime(t.period) < end_date - timedelta(days=90)]
                    
                    if recent and old:
                        recent_vol = sum([t.quantity_tons * 1000 for t in recent if t.quantity_tons])
                        old_vol = sum([t.quantity_tons * 1000 for t in old if t.quantity_tons])
                        features[f'{country}_volume_change'] = (recent_vol - old_vol) / old_vol if old_vol > 0 else 0
            
            return pd.DataFrame([features])
    
    def _extract_interaction_features(self, price_df: pd.DataFrame, 
                                    weather_df: pd.DataFrame,
                                    sentiment_df: pd.DataFrame, 
                                    trade_df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between data sources"""
        
        features = {}
        
        # Price-Weather interactions
        if not price_df.empty and not weather_df.empty:
            # Volatility during high risk
            if 'volatility_7d' in price_df.columns and 'max_regional_risk' in weather_df.columns:
                features['vol_risk_interaction'] = (
                    price_df['volatility_7d'].iloc[0] * 
                    weather_df['max_regional_risk'].iloc[0]
                )
            
            # Price momentum during weather events
            if 'momentum_10d' in price_df.columns and 'extreme_heat_days_30d' in weather_df.columns:
                features['momentum_weather_impact'] = (
                    price_df['momentum_10d'].iloc[0] * 
                    (1 + weather_df['extreme_heat_days_30d'].iloc[0] / 30)
                )
        
        # Sentiment-Price interactions
        if not sentiment_df.empty and not price_df.empty:
            # Sentiment-return alignment
            if 'sentiment_mean_7d' in sentiment_df.columns and 'return_7d' in price_df.columns:
                features['sentiment_price_alignment'] = (
                    sentiment_df['sentiment_mean_7d'].iloc[0] * 
                    np.sign(price_df['return_7d'].iloc[0])
                )
            
            # Sentiment volatility impact
            if 'sentiment_std' in sentiment_df.columns and 'volatility_30d' in price_df.columns:
                features['sentiment_vol_factor'] = (
                    sentiment_df['sentiment_std'].iloc[0] * 
                    price_df['volatility_30d'].iloc[0]
                )
        
        # Trade-Weather interactions
        if not trade_df.empty and not weather_df.empty:
            # Export impact during weather stress
            if 'total_export_volume' in trade_df.columns and 'overall_drought_risk' in weather_df.columns:
                features['trade_weather_stress'] = (
                    np.log1p(trade_df['total_export_volume'].iloc[0]) * 
                    weather_df['overall_drought_risk'].iloc[0]
                )
        
        # Multi-source risk score
        risk_components = []
        if not weather_df.empty and 'max_regional_risk' in weather_df.columns:
            risk_components.append(weather_df['max_regional_risk'].iloc[0])
        if not sentiment_df.empty and 'negative_ratio' in sentiment_df.columns:
            risk_components.append(sentiment_df['negative_ratio'].iloc[0])
        if not price_df.empty and 'volatility_30d' in price_df.columns:
            risk_components.append(min(1, price_df['volatility_30d'].iloc[0] * 10))
        
        if risk_components:
            features['composite_risk_score'] = np.mean(risk_components)
        
        return pd.DataFrame([features])
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately"""
        
        # Forward fill for time series continuity
        df = df.ffill(limit=3)
        
        # Fill remaining with median for numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Fill any remaining with 0
        df = df.fillna(0)
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features appropriately"""
        
        # Price returns are already in percentage
        return_cols = [col for col in df.columns if 'return' in col]
        
        # Risk scores are already 0-1
        risk_cols = [col for col in df.columns if 'risk' in col]
        
        # Sentiment is already -1 to 1
        sentiment_cols = [col for col in df.columns if 'sentiment' in col]
        
        # Scale other features
        for col in df.columns:
            if col not in return_cols + risk_cols + sentiment_cols:
                # Z-score normalization
                if df[col].std() > 0:
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        return df
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of a distribution"""
        if len(values) < 3:
            return 0
        
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0
        
        return np.mean([(x - mean) ** 3 for x in values]) / (std ** 3)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD indicators"""
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return (
            macd_line.iloc[-1] if not np.isnan(macd_line.iloc[-1]) else 0.0,
            signal_line.iloc[-1] if not np.isnan(signal_line.iloc[-1]) else 0.0,
            histogram.iloc[-1] if not np.isnan(histogram.iloc[-1]) else 0.0
        )
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            current_price = prices.iloc[-1]
            return current_price, current_price, current_price
        
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return (
            upper.iloc[-1] if not np.isnan(upper.iloc[-1]) else prices.iloc[-1],
            middle.iloc[-1] if not np.isnan(middle.iloc[-1]) else prices.iloc[-1],
            lower.iloc[-1] if not np.isnan(lower.iloc[-1]) else prices.iloc[-1]
        )
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        # Run a dummy extraction to get feature names
        dummy_date = datetime.now()
        features = self.extract_all_features(dummy_date, lookback_days=30)
        return list(features.columns) if not features.empty else []
    
    def create_feature_matrix(self, start_date: datetime, 
                            end_date: datetime,
                            frequency: str = 'D') -> pd.DataFrame:
        """
        Create feature matrix for a date range
        Useful for model training
        """
        logger.info(f"Creating feature matrix from {start_date} to {end_date}")
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        all_features = []
        for date in dates:
            try:
                features = self.extract_all_features(date)
                if not features.empty:
                    features['date'] = date
                    all_features.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract features for {date}: {e}")
        
        if all_features:
            feature_matrix = pd.concat(all_features, ignore_index=True)
            feature_matrix.set_index('date', inplace=True)
            logger.info(f"Created feature matrix with {len(feature_matrix)} samples and {len(feature_matrix.columns)} features")
            return feature_matrix
        else:
            logger.warning("No features extracted")
            return pd.DataFrame()


def demonstrate_feature_extraction():
    """Demonstrate the feature extraction pipeline"""
    print("🚀 Comprehensive Feature Extraction Pipeline")
    print("=" * 60)
    
    extractor = ComprehensiveFeatureExtractor()
    
    # Extract features for today
    target_date = datetime.now()
    features = extractor.extract_all_features(target_date)
    
    if not features.empty:
        print(f"\n📊 Extracted {len(features.columns)} features:")
        
        # Group features by category
        price_features = [f for f in features.columns if any(x in f for x in ['return', 'volatility', 'rsi', 'macd', 'momentum'])]
        weather_features = [f for f in features.columns if any(x in f for x in ['risk', 'temp', 'rainfall', 'extreme'])]
        sentiment_features = [f for f in features.columns if 'sentiment' in f or 'topic' in f or 'article' in f]
        trade_features = [f for f in features.columns if 'export' in f or 'trade' in f]
        interaction_features = [f for f in features.columns if 'interaction' in f or 'composite' in f]
        
        print(f"\n📈 Price Features ({len(price_features)}):")
        for f in price_features[:5]:
            print(f"  - {f}: {features[f].iloc[0]:.4f}")
        
        print(f"\n🌡️ Weather Features ({len(weather_features)}):")
        for f in weather_features[:5]:
            print(f"  - {f}: {features[f].iloc[0]:.4f}")
        
        print(f"\n📰 Sentiment Features ({len(sentiment_features)}):")
        for f in sentiment_features[:5]:
            print(f"  - {f}: {features[f].iloc[0]:.4f}")
        
        print(f"\n📦 Trade Features ({len(trade_features)}):")
        for f in trade_features[:5]:
            print(f"  - {f}: {features[f].iloc[0]:.4f}")
        
        print(f"\n🔗 Interaction Features ({len(interaction_features)}):")
        for f in interaction_features:
            print(f"  - {f}: {features[f].iloc[0]:.4f}")
        
        # Feature statistics
        print("\n📊 Feature Statistics:")
        print(f"  Non-zero features: {(features != 0).sum().sum()}/{features.size}")
        print(f"  Missing values: {features.isna().sum().sum()}")
        print(f"  Feature range: [{features.min().min():.2f}, {features.max().max():.2f}]")
        
    else:
        print("\n⚠️ No features extracted - check data availability")
    
    # Create feature matrix for modeling
    print("\n🔨 Creating feature matrix for modeling...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    feature_matrix = extractor.create_feature_matrix(start_date, end_date)
    
    if not feature_matrix.empty:
        print(f"✅ Feature matrix created:")
        print(f"   Shape: {feature_matrix.shape}")
        print(f"   Date range: {feature_matrix.index[0]} to {feature_matrix.index[-1]}")
        print(f"   Memory usage: {feature_matrix.memory_usage().sum() / 1024:.1f} KB")


if __name__ == "__main__":
    demonstrate_feature_extraction()