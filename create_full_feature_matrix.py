#!/usr/bin/env python3
"""
Create comprehensive feature matrix with all data sources
Including the newly analyzed sentiment data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features.comprehensive_feature_extractor import ComprehensiveFeatureExtractor
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def create_full_feature_matrix():
    """Create feature matrix for entire study period"""
    
    print("🔧 Creating comprehensive feature matrix...")
    print("=" * 60)
    
    extractor = ComprehensiveFeatureExtractor()
    
    # Define time period - last 365 days to ensure we have sentiment data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"📅 Time period: {start_date.date()} to {end_date.date()}")
    print("⏳ This will take a few minutes...\n")
    
    # Create feature matrix
    feature_matrix = extractor.create_feature_matrix(
        start_date=start_date,
        end_date=end_date,
        frequency='D'  # Daily features
    )
    
    if feature_matrix.empty:
        print("❌ Failed to create feature matrix")
        return None
    
    print(f"\n✅ Feature matrix created!")
    print(f"   Shape: {feature_matrix.shape}")
    print(f"   Features: {len(feature_matrix.columns)}")
    print(f"   Samples: {len(feature_matrix)}")
    
    # Analyze feature quality
    print("\n📊 Feature Quality Analysis:")
    
    # Group features by type
    price_features = [col for col in feature_matrix.columns if any(x in col for x in ['return', 'volatility', 'rsi', 'macd', 'momentum', 'sma', 'bb_'])]
    weather_features = [col for col in feature_matrix.columns if any(x in col for x in ['risk', 'temp', 'rainfall', 'extreme', 'weather'])]
    sentiment_features = [col for col in feature_matrix.columns if 'sentiment' in col or 'article' in col or any(x in col for x in ['positive', 'negative', 'topic'])]
    interaction_features = [col for col in feature_matrix.columns if 'interaction' in col or 'composite' in col]
    
    print(f"\n📈 Price Features: {len(price_features)}")
    print(f"🌡️  Weather Features: {len(weather_features)}")
    print(f"📰 Sentiment Features: {len(sentiment_features)}")
    print(f"🔗 Interaction Features: {len(interaction_features)}")
    
    # Check feature coverage
    print("\n📊 Feature Coverage:")
    for feature_group, features in [
        ("Price", price_features),
        ("Weather", weather_features),
        ("Sentiment", sentiment_features)
    ]:
        if features:
            # Calculate percentage of non-null values
            coverage = (feature_matrix[features].notna().sum().sum() / 
                       (len(features) * len(feature_matrix)) * 100)
            print(f"  {feature_group}: {coverage:.1f}% coverage")
            
            # Show sample features with values
            print(f"    Sample features:")
            for feat in features[:3]:
                mean_val = feature_matrix[feat].mean()
                std_val = feature_matrix[feat].std()
                print(f"      {feat}: mean={mean_val:.3f}, std={std_val:.3f}")
    
    # Save feature matrix
    output_file = 'data/processed/feature_matrix_complete.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    feature_matrix.to_csv(output_file)
    print(f"\n💾 Saved to: {output_file}")
    
    # Create target variable (future returns)
    print("\n🎯 Creating target variables...")
    
    # Load price data to create targets
    from sqlmodel import Session, select
    from app.core.database import engine
    from app.models.price_data import PriceData
    
    with Session(engine) as session:
        prices = session.exec(
            select(PriceData)
            .where(PriceData.date >= start_date.date())
            .where(PriceData.source == "Yahoo Finance")
            .order_by(PriceData.date)
        ).all()
        
        price_df = pd.DataFrame([
            {'date': p.date, 'price': p.price}
            for p in prices
        ])
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df.set_index('date', inplace=True)
        
        # Create multiple prediction horizons
        horizons = [1, 7, 30]  # 1 day, 1 week, 1 month
        
        for horizon in horizons:
            # Calculate future returns
            price_df[f'return_{horizon}d_future'] = price_df['price'].pct_change(horizon).shift(-horizon)
            
            # Binary classification (up/down)
            price_df[f'direction_{horizon}d_future'] = (price_df[f'return_{horizon}d_future'] > 0).astype(int)
        
        # Merge with features
        feature_matrix = feature_matrix.merge(
            price_df[[col for col in price_df.columns if 'future' in col]],
            left_index=True,
            right_index=True,
            how='left'
        )
    
    # Remove rows without targets
    feature_matrix = feature_matrix.dropna(subset=[f'return_{horizons[0]}d_future'])
    
    print(f"  Created targets for {len(horizons)} prediction horizons")
    print(f"  Final dataset shape: {feature_matrix.shape}")
    
    # Save complete dataset
    output_file_complete = 'data/processed/dataset_complete_with_targets.csv'
    feature_matrix.to_csv(output_file_complete)
    print(f"\n💾 Saved complete dataset to: {output_file_complete}")
    
    # Show target statistics
    print("\n🎯 Target Statistics:")
    for horizon in horizons:
        returns = feature_matrix[f'return_{horizon}d_future']
        directions = feature_matrix[f'direction_{horizon}d_future']
        
        print(f"\n  {horizon}-day horizon:")
        print(f"    Mean return: {returns.mean()*100:.2f}%")
        print(f"    Std return: {returns.std()*100:.2f}%")
        print(f"    % Positive: {directions.mean()*100:.1f}%")
        print(f"    Max return: {returns.max()*100:.1f}%")
        print(f"    Min return: {returns.min()*100:.1f}%")
    
    return feature_matrix

if __name__ == "__main__":
    feature_matrix = create_full_feature_matrix()
    
    if feature_matrix is not None:
        print("\n✅ Feature extraction complete!")
        print("📊 Ready for model training with:")
        print(f"   - {len(feature_matrix)} samples")
        print(f"   - {len([col for col in feature_matrix.columns if 'future' not in col])} features")
        print(f"   - {len([col for col in feature_matrix.columns if 'future' in col])} target variables")