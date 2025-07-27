#!/usr/bin/env python3
"""
Test Comtrade fetcher to see if it can get real data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline.comtrade_fetcher import ComtradeFetcher
import json

def test_comtrade_api():
    """Test if we can fetch real Comtrade data"""
    print("\n" + "="*60)
    print("TESTING UN COMTRADE API")
    print("="*60)
    
    fetcher = ComtradeFetcher()
    
    # Try to fetch just one month of data
    print("\nFetching sample data for testing...")
    
    # Check if the fetcher has a method to get data
    if hasattr(fetcher, 'fetch_export_data'):
        try:
            # Fetch minimal data
            data = fetcher.fetch_export_data(start_year=2024, end_year=2024)
            
            if data:
                print(f"\n✓ Successfully fetched {len(data)} records")
                # Show sample
                if isinstance(data, list) and data:
                    print("\nSample record:")
                    print(json.dumps(data[0], indent=2))
            else:
                print("\n❌ No data returned from API")
                
        except Exception as e:
            print(f"\n❌ Error fetching data: {str(e)}")
            print("\nThis might be because:")
            print("1. API requires authentication")
            print("2. Rate limits")
            print("3. Network issues")
    
    # Check if we already have the data locally
    print("\n" + "-"*60)
    print("Checking for existing Comtrade data files...")
    
    from pathlib import Path
    data_dir = Path("data/historical/trade")
    
    # Look for any CSV or detailed JSON files
    for pattern in ["*.csv", "comtrade_*.json", "export_data_*.json"]:
        files = list(data_dir.glob(pattern))
        if files:
            print(f"\nFound {pattern}:")
            for f in files:
                print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    test_comtrade_api()