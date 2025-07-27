#!/usr/bin/env python3
"""
Create complete dataset with features and targets
Properly handles date alignment and future returns
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features.comprehensive_feature_extractor import ComprehensiveFeatureExtractor
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlmodel import Session, select
from app.core.database import engine
from app.models.price_data import PriceData

def create_dataset_with_targets():
    """Create dataset with proper target variables"""
    
    print("🔧 Creating comprehensive dataset with targets...")
    print("=" * 60)
    
    extractor = ComprehensiveFeatureExtractor()
    
    # Get available price data range
    with Session(engine) as session:
        # Get min and max dates for price data
        oldest_price = session.scalar(
            select(PriceData.date)
            .where(PriceData.source == "Yahoo Finance")
            .order_by(PriceData.date)
            .limit(1)
        )
        
        newest_price = session.scalar(
            select(PriceData.date)
            .where(PriceData.source == "Yahoo Finance")
            .order_by(PriceData.date.desc())
            .limit(1)
        )
        
        if not oldest_price or not newest_price:
            print("❌ No price data found!")
            return None
            
        print(f"📅 Price data available: {oldest_price} to {newest_price}")
    
    # Define analysis period
    # We need to leave room for future returns (30 days)
    end_date = newest_price - timedelta(days=35)  # Leave room for 30-day future returns
    start_date = max(oldest_price + timedelta(days=90), end_date - timedelta(days=365))  # Need lookback data
    
    print(f"📊 Analysis period: {start_date} to {end_date}")
    print("⏳ Extracting features and creating targets...\n")
    
    # Create feature matrix
    all_data = []
    current_date = start_date
    
    while current_date <= end_date:
        try:
            # Extract features for this date (convert to datetime)
            features = extractor.extract_all_features(datetime.combine(current_date, datetime.min.time()), lookback_days=90)
            
            if not features.empty:
                # Get future returns for targets
                with Session(engine) as session:
                    # Get price on target date
                    current_price = session.scalar(
                        select(PriceData.price)
                        .where(PriceData.date == current_date)
                        .where(PriceData.source == "Yahoo Finance")
                    )
                    
                    if current_price:
                        # Calculate future returns
                        horizons = [1, 7, 30]
                        targets = {}
                        
                        for horizon in horizons:
                            future_date = current_date + timedelta(days=horizon)
                            future_price = session.scalar(
                                select(PriceData.price)
                                .where(PriceData.date == future_date)
                                .where(PriceData.source == "Yahoo Finance")
                            )
                            
                            if future_price:
                                # Calculate return
                                return_pct = (future_price - current_price) / current_price
                                targets[f'return_{horizon}d_future'] = return_pct
                                targets[f'direction_{horizon}d_future'] = 1 if return_pct > 0 else 0
                                targets[f'price_{horizon}d_future'] = future_price
                        
                        # Only add if we have all targets
                        if len(targets) == len(horizons) * 3:
                            # Combine features and targets
                            row_data = features.iloc[0].to_dict()
                            row_data.update(targets)
                            row_data['date'] = current_date
                            row_data['current_price'] = current_price
                            all_data.append(row_data)
                            
                            if len(all_data) % 50 == 0:
                                print(f"  Processed {len(all_data)} samples...")
        
        except Exception as e:
            print(f"  Warning: Failed to process {current_date}: {str(e)}")
        
        # Move to next day
        current_date += timedelta(days=1)
    
    if not all_data:
        print("❌ No valid samples created!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    df.set_index('date', inplace=True)
    
    print(f"\n✅ Dataset created!")
    print(f"   Total samples: {len(df)}")
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
        print(f"    Max return: {returns.max()*100:.1f}%")
        print(f"    Min return: {returns.min()*100:.1f}%")
    
    # Feature quality check
    print("\n📊 Feature Quality:")
    feature_cols = [c for c in df.columns if 'future' not in c and c != 'current_price']
    
    # Check for missing values
    missing_pct = (df[feature_cols].isna().sum() / len(df) * 100).sort_values(ascending=False)
    if missing_pct[missing_pct > 0].any():
        print("\n  Features with missing values:")
        for feat, pct in missing_pct[missing_pct > 0].head(10).items():
            print(f"    {feat}: {pct:.1f}%")
    else:
        print("  No missing values in features ✅")
    
    # Check for constant features
    constant_features = [f for f in feature_cols if df[f].nunique() <= 1]
    if constant_features:
        print(f"\n  Warning: {len(constant_features)} constant features found")
        for f in constant_features[:5]:
            print(f"    - {f}")
    
    # Save dataset
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete dataset
    output_file = os.path.join(output_dir, 'ml_dataset_complete.csv')
    df.to_csv(output_file)
    print(f"\n💾 Saved complete dataset to: {output_file}")
    
    # Save train/test split info
    split_date = df.index[-int(len(df) * 0.2)]  # 80/20 split
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Train: {len(train_df)} samples ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"   Test: {len(test_df)} samples ({test_df.index[0]} to {test_df.index[-1]})")
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'ml_dataset_train.csv'))
    test_df.to_csv(os.path.join(output_dir, 'ml_dataset_test.csv'))
    
    # Create feature importance preview
    print("\n🔍 Feature Preview (first 5 samples):")
    preview_features = ['return_7d', 'volatility_30d', 'sentiment_mean', 
                       'overall_drought_risk', 'composite_risk_score']
    
    available_preview = [f for f in preview_features if f in df.columns]
    if available_preview:
        print(df[available_preview + ['return_7d_future']].head())
    
    return df


if __name__ == "__main__":
    dataset = create_dataset_with_targets()
    
    if dataset is not None:
        print("\n✅ Dataset creation complete!")
        print("📊 Ready for model training")
        print("\n🚀 Next steps:")
        print("   1. Train multi-source prediction models")
        print("   2. Evaluate performance at different horizons")
        print("   3. Deploy best model to dashboard")