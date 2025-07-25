# XGBoost libomp Installation Fix Guide for macOS

## Problem Summary
XGBoost requires libomp (OpenMP library) for parallel computation, but macOS doesn't include OpenMP by default. This causes the error:
```
Library not loaded: /usr/local/opt/libomp/lib/libomp.dylib
```

## Solution 1: Homebrew Installation (Recommended)

### For Intel Macs:
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install libomp
brew install libomp

# Install XGBoost with pip
pip install xgboost

# If still having issues, reinstall XGBoost
pip uninstall xgboost
pip install --no-binary :all: xgboost
```

### For Apple Silicon Macs (M1/M2/M3):
```bash
# Ensure Homebrew is installed for ARM64
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install libomp
brew install libomp

# Set environment variables for compilation
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

# Install XGBoost
pip install xgboost
```

## Solution 2: Conda Installation (Alternative)

```bash
# Create a new conda environment
conda create -n xgboost-env python=3.9

# Activate the environment
conda activate xgboost-env

# Install XGBoost from conda-forge (includes libomp)
conda install -c conda-forge xgboost

# This automatically handles libomp dependencies
```

## Solution 3: Manual libomp Linking

If Homebrew installation doesn't work:

```bash
# Find where libomp is installed
find /usr/local -name "libomp.dylib" 2>/dev/null
find /opt/homebrew -name "libomp.dylib" 2>/dev/null

# Create symbolic links (adjust paths as needed)
# For Intel Macs:
sudo ln -s /usr/local/opt/libomp/lib/libomp.dylib /usr/local/lib/libomp.dylib

# For Apple Silicon:
sudo ln -s /opt/homebrew/opt/libomp/lib/libomp.dylib /usr/local/lib/libomp.dylib
```

## Solution 4: Install XGBoost with Pre-built Wheels

```bash
# Use pre-built wheels that include libomp
pip install xgboost --prefer-binary

# Or specify the wheel directly
# For Intel Mac:
pip install https://files.pythonhosted.org/packages/[specific-wheel-url]

# Check PyPI for the latest wheel URLs
```

## Verification Steps

1. **Test Import:**
```python
import xgboost as xgb
print(f"XGBoost version: {xgb.__version__}")
```

2. **Test Basic Functionality:**
```python
import xgboost as xgb
import numpy as np

# Create sample data
X = np.random.rand(100, 10)
y = np.random.randint(2, size=100)

# Create DMatrix
dtrain = xgb.DMatrix(X, label=y)

# Train a simple model
params = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}
model = xgb.train(params, dtrain, num_boost_round=10)

print("XGBoost is working correctly!")
```

3. **Check libomp Loading:**
```bash
# Check if libomp is properly linked
otool -L $(python -c "import xgboost; print(xgboost.__file__.replace('__init__.py', 'lib/libxgboost.dylib'))")
```

## Common Issues and Solutions

### Issue 1: "Library not loaded" after installation
```bash
# Set DYLD_LIBRARY_PATH (temporary fix)
export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib:$DYLD_LIBRARY_PATH

# For Apple Silicon:
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH
```

### Issue 2: Virtual Environment Issues
```bash
# Ensure you're installing in the correct environment
which python
which pip

# Activate your virtual environment first
source venv/bin/activate  # or conda activate your-env

# Then install
pip install xgboost
```

### Issue 3: Compilation Errors
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install gcc (if needed)
brew install gcc

# Set compiler flags
export CC=gcc-11  # or latest version
export CXX=g++-11
```

## Production Deployment Best Practices

1. **Use Docker:**
```dockerfile
FROM python:3.9-slim

# Install libomp
RUN apt-get update && apt-get install -y \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install XGBoost
RUN pip install xgboost
```

2. **Use Requirements File:**
```txt
# requirements.txt
xgboost==1.7.6  # Specify exact version
numpy>=1.20.0
pandas>=1.3.0
```

3. **Environment Setup Script:**
```bash
#!/bin/bash
# setup_xgboost.sh

echo "Setting up XGBoost environment..."

# Detect Mac architecture
if [[ $(uname -m) == 'arm64' ]]; then
    echo "Apple Silicon Mac detected"
    export HOMEBREW_PREFIX="/opt/homebrew"
else
    echo "Intel Mac detected"
    export HOMEBREW_PREFIX="/usr/local"
fi

# Install libomp if not present
if ! brew list libomp &>/dev/null; then
    echo "Installing libomp..."
    brew install libomp
fi

# Set environment variables
export LDFLAGS="-L${HOMEBREW_PREFIX}/opt/libomp/lib"
export CPPFLAGS="-I${HOMEBREW_PREFIX}/opt/libomp/include"
export DYLD_LIBRARY_PATH="${HOMEBREW_PREFIX}/opt/libomp/lib:$DYLD_LIBRARY_PATH"

# Install XGBoost
pip install xgboost

echo "XGBoost setup complete!"
```

## Testing Script

Save this as `test_xgboost.py`:

```python
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
```

## Summary

The most reliable approach is:
1. Install libomp via Homebrew
2. Set appropriate environment variables for your Mac architecture
3. Install XGBoost via pip with proper flags
4. Verify the installation works correctly

For production use, consider using Docker or conda environments to ensure consistent dependencies across different systems.