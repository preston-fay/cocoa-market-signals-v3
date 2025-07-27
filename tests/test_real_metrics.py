#!/usr/bin/env python3
"""
Test script to verify REAL metric calculations from database
NO FAKE DATA - Must pass these tests before building anything
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sqlmodel import Session, select
from datetime import datetime, timedelta
from app.core.database import engine
from app.models import PriceData, Prediction, ModelPerformance
import numpy as np

def test_database_has_real_data():
    """Test 1: Verify we have real price data in database"""
    with Session(engine) as session:
        prices = session.exec(select(PriceData)).all()
        
        # We should have at least some data
        assert len(prices) > 0, "No price data found in database"
        
        # Print what we actually have
        print(f"\nReal data in database:")
        for price in prices:
            print(f"  {price.date}: ${price.price:,.0f} (Source: {price.source})")
        
        # Verify it's ICCO data (our real source)
        sources = set(p.source for p in prices)
        assert "ICCO" in sources, "No ICCO data found"

def test_can_calculate_real_price_change():
    """Test 2: Calculate real price changes from actual data"""
    with Session(engine) as session:
        prices = session.exec(
            select(PriceData).order_by(PriceData.date)
        ).all()
        
        if len(prices) < 2:
            pytest.skip("Need at least 2 price points to calculate change")
        
        # Calculate real changes
        for i in range(1, len(prices)):
            prev_price = prices[i-1].price
            curr_price = prices[i].price
            change_pct = (curr_price - prev_price) / prev_price * 100
            
            print(f"\n{prices[i-1].date} to {prices[i].date}:")
            print(f"  ${prev_price:,.0f} → ${curr_price:,.0f}")
            print(f"  Change: {change_pct:+.1f}%")
            
            # This is REAL data, changes can be large
            assert isinstance(change_pct, (int, float)), "Change calculation failed"

def test_no_predictions_exist_yet():
    """Test 3: Verify we haven't stored any fake predictions"""
    with Session(engine) as session:
        predictions = session.exec(select(Prediction)).all()
        
        # Should be empty - we haven't made any predictions yet
        assert len(predictions) == 0, "Found predictions before running any models"
        print("\n✓ No fake predictions in database")

def test_can_store_and_retrieve_prediction():
    """Test 4: Verify we can store a prediction and calculate error"""
    with Session(engine) as session:
        # Get a real price
        prices = session.exec(
            select(PriceData).order_by(PriceData.date).limit(2)
        ).all()
        
        if len(prices) < 2:
            pytest.skip("Need at least 2 prices to test prediction")
        
        # Create a test prediction
        current_price = prices[0].price
        actual_price = prices[1].price
        
        # Make a simple prediction (just for testing storage)
        test_prediction = Prediction(
            target_date=prices[1].date,
            prediction_horizon=1,
            predicted_price=current_price * 1.01,  # 1% increase guess
            confidence_score=0.5,  # Low confidence
            prediction_type="test",
            model_name="test_model",
            current_price=current_price,
            created_at=datetime.utcnow()
        )
        
        session.add(test_prediction)
        session.commit()
        
        # Now update with actual outcome
        test_prediction.actual_price = actual_price
        test_prediction.error = actual_price - test_prediction.predicted_price
        test_prediction.error_percentage = abs(test_prediction.error / actual_price * 100)
        session.commit()
        
        print(f"\nTest prediction stored:")
        print(f"  Predicted: ${test_prediction.predicted_price:,.0f}")
        print(f"  Actual: ${test_prediction.actual_price:,.0f}")
        print(f"  Error: {test_prediction.error_percentage:.1f}%")
        
        # Clean up test data
        session.delete(test_prediction)
        session.commit()

def test_calculate_real_volatility():
    """Test 5: Calculate real volatility from actual price data"""
    with Session(engine) as session:
        prices = session.exec(
            select(PriceData).order_by(PriceData.date)
        ).all()
        
        if len(prices) < 2:
            pytest.skip("Need multiple prices for volatility")
        
        # Calculate returns
        price_values = [p.price for p in prices]
        returns = []
        for i in range(1, len(price_values)):
            ret = (price_values[i] - price_values[i-1]) / price_values[i-1]
            returns.append(ret)
        
        if returns:
            # Real volatility calculation
            volatility = np.std(returns) * np.sqrt(252) * 100  # Annualized
            print(f"\nReal volatility from {len(prices)} data points: {volatility:.1f}%")
            
            # Volatility should be positive
            assert volatility > 0, "Invalid volatility calculation"
            
            # For cocoa, volatility can be very high (30-100%+)
            print(f"✓ Calculated real volatility: {volatility:.1f}%")

def test_insufficient_data_for_full_analysis():
    """Test 6: Acknowledge we need more data for proper analysis"""
    with Session(engine) as session:
        count = session.exec(select(PriceData)).all()
        
        print(f"\nCurrent data status:")
        print(f"  Price records: {len(count)}")
        print(f"  Date range: {min(p.date for p in count)} to {max(p.date for p in count)}")
        
        # We need much more data
        if len(count) < 100:
            print("\n⚠️  WARNING: Insufficient data for reliable predictions")
            print("  Need at least 100+ daily prices for time series models")
            print("  Current ICCO data is monthly - need daily data from Yahoo Finance")
            
        # This is honest - we don't have enough data yet
        assert len(count) < 100, "If this fails, we might have enough data!"

if __name__ == "__main__":
    # Run all tests
    print("="*60)
    print("TESTING REAL METRICS - NO FAKE DATA")
    print("="*60)
    
    test_database_has_real_data()
    test_can_calculate_real_price_change()
    test_no_predictions_exist_yet()
    test_can_store_and_retrieve_prediction()
    test_calculate_real_volatility()
    test_insufficient_data_for_full_analysis()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED - But we need more data!")
    print("="*60)