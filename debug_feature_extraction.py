#!/usr/bin/env python3
"""
Debug feature extraction to find datetime comparison issues
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features.comprehensive_feature_extractor import ComprehensiveFeatureExtractor
from datetime import datetime, timedelta
import traceback

def debug_extraction():
    """Debug single date extraction"""
    
    extractor = ComprehensiveFeatureExtractor()
    
    # Test with a single date
    test_date = datetime(2023, 10, 24)
    
    print(f"Testing feature extraction for {test_date}")
    print("=" * 60)
    
    try:
        features = extractor.extract_all_features(test_date, lookback_days=90)
        print(f"Success! Extracted {len(features.columns) if features is not None else 0} features")
        if features is not None and not features.empty:
            print(f"Feature shape: {features.shape}")
            print(f"Sample features: {list(features.columns[:5])}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Try to identify the exact location
        import pdb
        import sys
        
        # Get the traceback
        tb = sys.exc_info()[2]
        
        # Walk through the traceback
        while tb is not None:
            frame = tb.tb_frame
            filename = frame.f_code.co_filename
            if 'comprehensive_feature_extractor.py' in filename:
                print(f"\nError in {filename}:")
                print(f"  Line {tb.tb_lineno}: {frame.f_code.co_name}")
                # Print local variables at the error point
                for var_name, var_value in frame.f_locals.items():
                    if 'date' in var_name.lower():
                        print(f"    {var_name} = {var_value} (type: {type(var_value)})")
            tb = tb.tb_next


if __name__ == "__main__":
    debug_extraction()