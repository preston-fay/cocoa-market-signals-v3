#!/usr/bin/env python3
"""
Test what data files we actually have available
"""
import json
from pathlib import Path

def test_available_data_files():
    """Check what data files exist in the project"""
    print("\n" + "="*60)
    print("CHECKING AVAILABLE DATA FILES")
    print("="*60)
    
    data_dir = Path("data/historical")
    
    # Check each data type
    for subdir in ["prices", "trade", "weather", "economics"]:
        dir_path = data_dir / subdir
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            print(f"\n{subdir.upper()} ({len(files)} files):")
            for f in sorted(files):
                print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
                
                # If it's JSON, show a preview
                if f.suffix == ".json" and f.stat().st_size > 0:
                    try:
                        with open(f) as fh:
                            data = json.load(fh)
                            if isinstance(data, dict):
                                print(f"    Keys: {', '.join(list(data.keys())[:5])}")
                            elif isinstance(data, list):
                                print(f"    Items: {len(data)}")
                    except:
                        print(f"    [Could not parse JSON]")

def test_check_csv_files():
    """Check for any CSV files that might contain data"""
    print("\n" + "="*60)
    print("CHECKING FOR CSV FILES")
    print("="*60)
    
    # Look for CSV files anywhere in data directory
    csv_files = list(Path("data").rglob("*.csv"))
    
    if csv_files:
        print(f"\nFound {len(csv_files)} CSV files:")
        for f in csv_files:
            print(f"  - {f} ({f.stat().st_size / 1024:.1f} KB)")
    else:
        print("\nNo CSV files found in data directory")
        print("Note: We may need to generate CSV exports from JSON data")

def test_comtrade_actual_data():
    """Check if we have actual Comtrade data beyond metadata"""
    print("\n" + "="*60)
    print("CHECKING FOR ACTUAL COMTRADE DATA")
    print("="*60)
    
    # The metadata says we have 36 records - where are they?
    trade_dir = Path("data/historical/trade")
    
    # Look for files that might contain the actual data
    for f in trade_dir.glob("*"):
        if f.name != "export_data_metadata.json":
            print(f"\nFound: {f.name}")
            if f.suffix == ".json":
                with open(f) as fh:
                    data = json.load(fh)
                    print(f"  Type: {type(data).__name__}")
                    if isinstance(data, list):
                        print(f"  Records: {len(data)}")
                        if data:
                            print(f"  Sample record keys: {list(data[0].keys())}")
    
    # Check if the data might be embedded in the metadata file
    metadata_file = trade_dir / "export_data_metadata.json"
    with open(metadata_file) as f:
        metadata = json.load(f)
        if "data" in metadata or "records" in metadata:
            print("\n✓ Found data embedded in metadata file")
        else:
            print("\n❌ No actual trade data found - only metadata")
            print("   The metadata mentions 36 records but they're not here")

def test_weather_actual_data():
    """Check if we have actual weather data beyond metadata"""
    print("\n" + "="*60)
    print("CHECKING FOR ACTUAL WEATHER DATA")  
    print("="*60)
    
    weather_dir = Path("data/historical/weather")
    
    # Check the summary file
    summary_file = weather_dir / "weather_summary_2yr.json"
    if summary_file.exists():
        with open(summary_file) as f:
            data = json.load(f)
            print(f"\nWeather summary file:")
            print(f"  Top-level keys: {list(data.keys())}")
            
            # Check if it has actual data
            if "locations" in data:
                locs = data["locations"]
                if isinstance(locs, dict) and locs:
                    first_loc = list(locs.keys())[0]
                    print(f"  Locations: {len(locs)}")
                    print(f"  Sample location ({first_loc}): {list(locs[first_loc].keys())[:5]}")
                    
                    # Check if there's daily data
                    if "daily_data" in locs[first_loc]:
                        daily = locs[first_loc]["daily_data"]
                        print(f"  Daily records for {first_loc}: {len(daily)}")
                    else:
                        print(f"  No daily_data key found")

if __name__ == "__main__":
    test_available_data_files()
    test_check_csv_files()
    test_comtrade_actual_data()
    test_weather_actual_data()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nWe need to check if the actual data exists in:")
    print("1. Separate CSV files")
    print("2. Embedded in the JSON files") 
    print("3. Needs to be fetched from APIs")
    print("="*60)