#!/usr/bin/env python3
"""
Create FULL dataset using ALL available data
Properly handles all date ranges and missing values
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features.comprehensive_feature_extractor import ComprehensiveFeatureExtractor
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlmodel import Session, select, func
from app.core.database import engine
from app.models.price_data import PriceData
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_full_dataset():
    """Create dataset using ALL available price data"""
    
    print("🔧 Creating FULL dataset with ALL available data...")
    print("=" * 60)
    
    extractor = ComprehensiveFeatureExtractor()
    
    # Get actual data range from database
    with Session(engine) as session:
        # Get price data range
        price_dates = session.exec(
            select(PriceData.date)
            .where(PriceData.source == "Yahoo Finance")
            .order_by(PriceData.date)
        ).all()
        
        if not price_dates:
            print("❌ No price data found!")
            return None
            
        print(f"📅 Found {len(price_dates)} days of price data")
        print(f"   Date range: {price_dates[0]} to {price_dates[-1]}")
    
    # Create dataset for all dates where we can calculate features
    # Need at least 90 days of history for feature calculation
    start_idx = 90  # Start after we have enough history
    end_idx = len(price_dates) - 35  # Leave room for 30-day future returns
    
    usable_dates = price_dates[start_idx:end_idx]
    print(f"\n🎯 Creating features for {len(usable_dates)} dates")
    print(f"   Analysis period: {usable_dates[0]} to {usable_dates[-1]}")
    
    # Process in batches for better performance
    all_data = []
    batch_size = 50
    
    for i in range(0, len(usable_dates), batch_size):
        batch_dates = usable_dates[i:i+batch_size]
        print(f"\n📊 Processing batch {i//batch_size + 1}/{(len(usable_dates)-1)//batch_size + 1}")
        
        for date in batch_dates:
            try:
                # Convert date to datetime for feature extraction
                target_datetime = datetime.combine(date, datetime.min.time())
                
                # Extract features
                features = extractor.extract_all_features(target_datetime, lookback_days=90)
                
                if not features.empty:
                    # Get current and future prices
                    with Session(engine) as session:
                        # Current price
                        current_price = session.scalar(
                            select(PriceData.price)
                            .where(PriceData.date == date)
                            .where(PriceData.source == "Yahoo Finance")
                        )
                        
                        if current_price:
                            # Calculate future returns for different horizons
                            row_data = features.iloc[0].to_dict()
                            row_data['date'] = date
                            row_data['current_price'] = current_price
                            
                            # Get future prices and calculate returns
                            horizons = [1, 7, 30]
                            has_all_targets = True
                            
                            for horizon in horizons:
                                future_date = date + timedelta(days=horizon)
                                future_price = session.scalar(
                                    select(PriceData.price)
                                    .where(PriceData.date == future_date)
                                    .where(PriceData.source == "Yahoo Finance")
                                )
                                
                                if future_price:
                                    return_pct = (future_price - current_price) / current_price
                                    row_data[f'return_{horizon}d_future'] = return_pct
                                    row_data[f'direction_{horizon}d_future'] = 1 if return_pct > 0 else 0
                                    row_data[f'price_{horizon}d_future'] = future_price
                                else:
                                    has_all_targets = False
                                    break
                            
                            # Only add if we have all targets
                            if has_all_targets:
                                all_data.append(row_data)
                                
                                if len(all_data) % 10 == 0:
                                    print(f"  ✓ Processed {len(all_data)} valid samples")
            
            except Exception as e:
                # Skip dates with errors (weekends, holidays, etc.)
                continue
        
        print(f"  Batch complete. Total samples: {len(all_data)}")
    
    if not all_data:
        print("❌ No valid samples created!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"\n✅ FULL dataset created!")
    print(f"   Total samples: {len(df)}")
    print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Features: {len([c for c in df.columns if 'future' not in c and c != 'current_price'])}")
    print(f"   Targets: {len([c for c in df.columns if 'future' in c])}")
    
    # Analyze targets
    print("\n🎯 Target Statistics:")
    for horizon in [1, 7, 30]:
        returns = df[f'return_{horizon}d_future']
        directions = df[f'direction_{horizon}d_future']
        
        print(f"\n  {horizon}-day horizon:")
        print(f"    Mean return: {returns.mean()*100:.2f}%")
        print(f"    Std return: {returns.std()*100:.2f}%")
        print(f"    % Positive: {directions.mean()*100:.1f}%")
        print(f"    Sharpe ratio: {np.sqrt(252/horizon) * returns.mean() / returns.std():.2f}")
    
    # Feature coverage analysis
    print("\n📊 Feature Coverage:")
    feature_cols = [c for c in df.columns if 'future' not in c and c != 'current_price']
    
    # Group features
    price_features = [f for f in feature_cols if any(x in f for x in ['return', 'volatility', 'rsi', 'macd'])]
    weather_features = [f for f in feature_cols if any(x in f for x in ['risk', 'temp', 'rainfall'])]
    sentiment_features = [f for f in feature_cols if any(x in f for x in ['sentiment', 'article'])]
    
    for group_name, features in [
        ("Price", price_features),
        ("Weather", weather_features),
        ("Sentiment", sentiment_features)
    ]:
        if features:
            coverage = df[features].notna().sum().sum() / (len(features) * len(df)) * 100
            print(f"  {group_name}: {len(features)} features, {coverage:.1f}% coverage")
    
    # Save dataset
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete dataset
    output_file = os.path.join(output_dir, 'full_ml_dataset.csv')
    df.to_csv(output_file)
    print(f"\n💾 Saved complete dataset to: {output_file}")
    
    # Create train/test splits
    # Use time-based split (80/20)
    split_date = df.index[int(len(df) * 0.8)]
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Train: {len(train_df)} samples ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"   Test: {len(test_df)} samples ({test_df.index[0].date()} to {test_df.index[-1].date()})")
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'full_ml_train.csv'))
    test_df.to_csv(os.path.join(output_dir, 'full_ml_test.csv'))
    
    return df


if __name__ == "__main__":
    dataset = create_full_dataset()
    
    if dataset is not None:
        print("\n✅ Full dataset creation complete!")
        print(f"🚀 Ready for advanced modeling with {len(dataset)} samples")