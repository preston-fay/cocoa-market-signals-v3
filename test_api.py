"""Test API endpoint directly"""
from src.dashboard.app_realtime import (
    load_data, get_month_data, get_data_at_date, 
    make_predictions_at_date, calculate_metrics_at_date
)
import pandas as pd

try:
    # Load data
    df, backtest_results = load_data()
    print(f"✓ Loaded {len(df)} rows of data")
    
    # Test specific month
    year_month = "2024-03"
    selected_date = pd.Period(year_month, 'M').to_timestamp('M')
    print(f"\n✓ Testing month: {year_month}")
    print(f"  Selected date: {selected_date}")
    
    # Get historical data
    df_historical = get_data_at_date(df, selected_date)
    print(f"  Historical data: {len(df_historical)} rows")
    
    # Calculate metrics
    metrics = calculate_metrics_at_date(df_historical)
    print(f"\n✓ Metrics calculated:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Test predictions
    from src.backtesting.time_aware_backtest import TimeAwareBacktester
    backtester = TimeAwareBacktester()
    predictions = make_predictions_at_date(df_historical, selected_date, backtester)
    
    if predictions:
        print(f"\n✓ Predictions generated for {len(predictions)} windows")
    else:
        print("\n✗ No predictions generated")
        
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()