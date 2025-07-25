#!/usr/bin/env python3
"""Test XGBoost installation"""

import sys
import subprocess

def test_xgboost():
    try:
        import xgboost as xgb
        print(f"✓ XGBoost {xgb.__version__} imported successfully")
        
        # Test basic functionality
        import numpy as np
        X = np.random.rand(100, 5)
        y = np.random.randint(2, size=100)
        
        dtrain = xgb.DMatrix(X, label=y)
        params = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}
        model = xgb.train(params, dtrain, num_boost_round=10, verbose_eval=False)
        
        print("✓ XGBoost training works correctly")
        print("✓ libomp is properly linked")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import XGBoost: {e}")
        print("\nTry running:")
        print("  brew install libomp")
        print("  pip install xgboost")
        return False
        
    except Exception as e:
        print(f"✗ XGBoost test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_xgboost()
    sys.exit(0 if success else 1)