#!/bin/bash
# setup_xgboost.sh - Comprehensive XGBoost setup for macOS

echo "Setting up XGBoost environment..."
echo "================================="

# Detect Mac architecture
if [[ $(uname -m) == 'arm64' ]]; then
    echo "✓ Apple Silicon Mac detected"
    export HOMEBREW_PREFIX="/opt/homebrew"
else
    echo "✓ Intel Mac detected"
    export HOMEBREW_PREFIX="/usr/local"
fi

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "✗ Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✓ Homebrew is installed"
fi

# Install libomp if not present
if ! brew list libomp &>/dev/null; then
    echo "Installing libomp..."
    brew install libomp
else
    echo "✓ libomp is already installed"
fi

# Set environment variables
echo ""
echo "Setting environment variables..."
export LDFLAGS="-L${HOMEBREW_PREFIX}/opt/libomp/lib"
export CPPFLAGS="-I${HOMEBREW_PREFIX}/opt/libomp/include"
export DYLD_LIBRARY_PATH="${HOMEBREW_PREFIX}/opt/libomp/lib:$DYLD_LIBRARY_PATH"

echo "LDFLAGS=$LDFLAGS"
echo "CPPFLAGS=$CPPFLAGS"
echo "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"

# Check Python environment
echo ""
echo "Checking Python environment..."
python3 --version

# Create virtual environment if it doesn't exist
VENV_DIR="xgboost_env"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install XGBoost
echo ""
echo "Installing XGBoost..."
pip install xgboost

# Test the installation
echo ""
echo "Testing XGBoost installation..."
python3 -c "
import xgboost as xgb
import numpy as np
print(f'✓ XGBoost {xgb.__version__} imported successfully')
X = np.random.rand(10, 5)
y = np.random.randint(2, size=10)
dtrain = xgb.DMatrix(X, label=y)
print('✓ XGBoost DMatrix created successfully')
print('✓ Installation successful!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================="
    echo "XGBoost setup complete!"
    echo ""
    echo "To use XGBoost in the future:"
    echo "1. Activate the virtual environment: source $VENV_DIR/bin/activate"
    echo "2. Import XGBoost: import xgboost as xgb"
    echo ""
    echo "For persistent environment variables, add these to your ~/.zshrc or ~/.bash_profile:"
    echo "export LDFLAGS=\"-L${HOMEBREW_PREFIX}/opt/libomp/lib\""
    echo "export CPPFLAGS=\"-I${HOMEBREW_PREFIX}/opt/libomp/include\""
    echo "export DYLD_LIBRARY_PATH=\"${HOMEBREW_PREFIX}/opt/libomp/lib:\$DYLD_LIBRARY_PATH\""
else
    echo ""
    echo "✗ XGBoost installation failed!"
    echo "Please check the error messages above."
    echo ""
    echo "Common fixes:"
    echo "1. Ensure Xcode Command Line Tools are installed: xcode-select --install"
    echo "2. Try reinstalling libomp: brew reinstall libomp"
    echo "3. Check the troubleshooting guide in docs/xgboost_libomp_fix_guide.md"
    exit 1
fi