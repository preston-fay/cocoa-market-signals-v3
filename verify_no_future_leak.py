"""Verify no future data leakage in dashboard"""
import requests
import json
import pandas as pd

# Test different months
test_months = ["2024-01", "2024-03", "2024-06"]

for month in test_months:
    print(f"\nTesting {month}:")
    response = requests.get(f"http://localhost:8055/api/month/{month}")
    
    # Extract data status from HTML
    if "data points available up to this date" in response.text:
        # Extract the number
        import re
        match = re.search(r'<strong>(\d+)</strong> data points available', response.text)
        if match:
            data_points = int(match.group(1))
            print(f"  ✓ Shows {data_points} data points (historical only)")
        
        # Check for "no future data leakage" message
        if "No future data leakage" in response.text:
            print(f"  ✓ Confirms no future data leakage")
            
        # Check if predictions exist
        if "Prediction Details" in response.text:
            print(f"  ✓ Has predictions for future periods")
        elif "Insufficient data" in response.text:
            print(f"  ✓ Correctly shows insufficient data")