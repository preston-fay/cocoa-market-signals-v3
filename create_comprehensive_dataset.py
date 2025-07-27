#!/usr/bin/env python3
"""
Create COMPREHENSIVE dataset using ALL available historical data
Processes entire 2023-2025 period with proper error handling
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

def create_comprehensive_dataset():
    """Create dataset using ALL available historical data (2023-2025)"""
    
    print("🔧 Creating COMPREHENSIVE dataset with ALL historical data...")
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
        
        # Count by year
        years = {}
        for date in price_dates:
            year = date.year
            years[year] = years.get(year, 0) + 1
        
        print("\n📊 Price data by year:")
        for year, count in sorted(years.items()):
            print(f"   {year}: {count} records")
    
    # Process ALL dates with enough history
    # Start from October 2023 (90 days after data starts)
    start_date = datetime(2023, 10, 24)  # ~90 days after 2023-07-26
    end_date = datetime(2025, 6, 20)     # ~35 days before 2025-07-25
    
    # Get all dates in range
    all_dates = []
    current_date = start_date
    while current_date <= end_date:
        # Check if we have price data for this date
        if current_date.date() in price_dates:
            all_dates.append(current_date.date())
        current_date += timedelta(days=1)
    
    print(f"\n🎯 Processing {len(all_dates)} potential dates")
    print(f"   Analysis period: {all_dates[0]} to {all_dates[-1]}")
    
    # Process all dates with detailed error tracking
    all_data = []
    failed_dates = []
    success_count = 0
    
    for i, date in enumerate(all_dates):
        if i % 50 == 0:
            print(f"\n📊 Progress: {i}/{len(all_dates)} dates processed")
            print(f"   Successful: {success_count}, Failed: {len(failed_dates)}")
        
        try:
            # Convert date to datetime for feature extraction
            target_datetime = datetime.combine(date, datetime.min.time())
            
            # Extract features with error handling
            features = extractor.extract_all_features(target_datetime, lookback_days=90)
            
            if features is not None and not features.empty:
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
                            success_count += 1
                        else:
                            failed_dates.append((date, "Missing future prices"))
                    else:
                        failed_dates.append((date, "No current price"))
            else:
                failed_dates.append((date, "Feature extraction failed"))
                            
        except Exception as e:
            failed_dates.append((date, str(e)))
            continue
    
    print(f"\n📊 Final processing results:")
    print(f"   Total dates processed: {len(all_dates)}")
    print(f"   Successful samples: {len(all_data)}")
    print(f"   Failed dates: {len(failed_dates)}")
    
    if failed_dates:
        # Analyze failure reasons
        failure_reasons = {}
        for date, reason in failed_dates:
            short_reason = reason.split(':')[0] if ':' in reason else reason
            failure_reasons[short_reason] = failure_reasons.get(short_reason, 0) + 1
        
        print("\n❌ Failure analysis:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"   {reason}: {count} occurrences")
    
    if not all_data:
        print("\n❌ No valid samples created!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"\n✅ COMPREHENSIVE dataset created!")
    print(f"   Total samples: {len(df)}")
    print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Analyze data distribution by year
    print("\n📊 Samples by year:")
    for year in sorted(df.index.year.unique()):
        year_data = df[df.index.year == year]
        print(f"   {year}: {len(year_data)} samples")
    
    # Feature analysis
    feature_cols = [c for c in df.columns if 'future' not in c and c != 'current_price']
    print(f"\n📈 Features: {len(feature_cols)}")
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
    print("\n📊 Feature Coverage by Group:")
    feature_groups = {
        'Price': [f for f in feature_cols if any(x in f for x in ['return', 'volatility', 'rsi', 'macd', 'momentum'])],
        'Weather': [f for f in feature_cols if any(x in f for x in ['risk', 'temp', 'rainfall', 'drought', 'flood'])],
        'Sentiment': [f for f in feature_cols if any(x in f for x in ['sentiment', 'article', 'positive', 'negative'])],
        'Trade': [f for f in feature_cols if any(x in f for x in ['export', 'import', 'volume', 'trade'])]
    }
    
    for group_name, features in feature_groups.items():
        if features:
            coverage = df[features].notna().sum().sum() / (len(features) * len(df)) * 100
            print(f"  {group_name}: {len(features)} features, {coverage:.1f}% coverage")
    
    # Save dataset
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save complete dataset
    output_file = os.path.join(output_dir, 'comprehensive_ml_dataset.csv')
    df.to_csv(output_file)
    print(f"\n💾 Saved comprehensive dataset to: {output_file}")
    
    # Create train/test splits (time-based)
    split_date = df.index[int(len(df) * 0.8)]
    train_df = df[df.index < split_date]
    test_df = df[df.index >= split_date]
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Train: {len(train_df)} samples ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"   Test: {len(test_df)} samples ({test_df.index[0].date()} to {test_df.index[-1].date()})")
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'comprehensive_train.csv'))
    test_df.to_csv(os.path.join(output_dir, 'comprehensive_test.csv'))
    
    return df


if __name__ == "__main__":
    dataset = create_comprehensive_dataset()
    
    if dataset is not None:
        print("\n✅ Comprehensive dataset creation complete!")
        print(f"🚀 Ready for advanced modeling with {len(dataset)} samples across {dataset.index[-1].year - dataset.index[0].year + 1} years")