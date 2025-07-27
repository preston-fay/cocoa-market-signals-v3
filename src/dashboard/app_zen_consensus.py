#!/usr/bin/env python3
"""
Zen Consensus Dashboard - Actual vs Predicted Prices
Following ALL Kearney Design Standards
Dark theme ONLY - NO red/green/yellow/blue colors
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

from ..models.zen_orchestrator import ZenOrchestrator
from ..models.multi_source_signal_detector import MultiSourceSignalDetector
from ..data_pipeline.unified_pipeline_real import UnifiedDataPipeline

# Initialize FastAPI
app = FastAPI(title="Cocoa Market Signals - Zen Consensus Dashboard")

# Configure templates
templates = Jinja2Templates(directory="templates")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kearney Color Standards (from preston-dev-setup)
KEARNEY_COLORS = {
    'primary_purple': '#6f42c1',
    'secondary_purple': '#9955bb',
    'primary_charcoal': '#272b30',
    'secondary_charcoal': '#3a3f44',
    'white': '#FFFFFF',
    'light_gray': '#e9ecef',
    'medium_gray': '#999999',
    'dark_gray': '#52575c',
    'border_gray': '#7a8288'
}

# Global data cache
_data_cache = {
    'df': None,
    'last_update': None,
    'consensus_result': None,
    'signals': None
}

def load_or_create_data():
    """Load real data or create sample data"""
    try:
        # Try to load real data
        pipeline = UnifiedDataPipeline()
        df = pipeline.prepare_unified_dataset()
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.error(f"Error loading real data: {e}")
    
    # Create sample data if real data not available
    dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
    np.random.seed(42)
    
    # Realistic price movement
    base_price = 7573
    volatility = 2533
    returns = np.random.normal(0, volatility/base_price/np.sqrt(252), len(dates))
    
    prices = [base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        new_price = max(3282, min(12565, new_price))
        prices.append(new_price)
    
    # Recent trend to current price
    prices[-30:] = np.linspace(prices[-30], 8440, 30)
    
    df = pd.DataFrame({
        'price': prices,
        'volume': np.random.uniform(1000, 5000, len(dates)),
        'trade_volume_change': np.random.normal(0, 0.1, len(dates)),
        'avg_precipitation': np.random.normal(50, 20, len(dates)),
        'avg_temperature': np.random.normal(25, 3, len(dates))
    }, index=dates)
    
    return df

def update_predictions():
    """Update predictions using Zen Consensus"""
    global _data_cache
    
    # Load data
    df = load_or_create_data()
    _data_cache['df'] = df
    
    # Run Zen Consensus
    orchestrator = ZenOrchestrator()
    consensus_result = orchestrator.run_zen_consensus(df)
    _data_cache['consensus_result'] = consensus_result
    
    # Detect signals
    detector = MultiSourceSignalDetector()
    signals = detector.detect_all_signals(df)
    _data_cache['signals'] = signals
    
    # Run backtest for historical predictions
    backtest = orchestrator.backtest_zen_consensus(df, test_days=90)
    _data_cache['backtest'] = backtest
    
    _data_cache['last_update'] = datetime.now()
    
    logger.info("Predictions updated successfully")

@app.on_event("startup")
async def startup_event():
    """Initialize data on startup"""
    update_predictions()

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    
    # Get latest data
    df = _data_cache['df']
    consensus = _data_cache['consensus_result']['consensus']
    signals = _data_cache['signals']
    backtest = _data_cache['backtest']
    
    # Prepare chart data
    recent_days = 90
    recent_df = df.iloc[-recent_days:]
    
    # Actual prices
    actual_prices = recent_df['price'].tolist()
    dates = [d.strftime('%Y-%m-%d') for d in recent_df.index]
    
    # Get predictions from backtest
    backtest_df = backtest['results_df']
    predicted_prices = backtest_df['consensus_forecasts'].tolist()
    
    # Calculate performance metrics
    current_price = df['price'].iloc[-1]
    forecast_price = consensus['consensus_forecast']
    price_change_pct = (forecast_price - current_price) / current_price * 100
    
    # Signal summary
    signal_summary = signals['summary']
    
    # Format recommendations
    recommendations = [
        {'icon': rec.split()[0], 'text': ' '.join(rec.split()[1:])}
        for rec in consensus['recommendations']
    ]
    
    # Role contributions for display
    role_contributions = []
    for role, contrib in consensus['role_contributions'].items():
        role_contributions.append({
            'role': role.replace('_', ' ').title(),
            'prediction': f"${contrib['prediction']:,.0f}",
            'change_pct': f"{contrib['change_pct']:+.1f}%",
            'confidence': f"{contrib['confidence']*100:.0f}%"
        })
    
    # Market context
    context = consensus['market_context']
    
    # Prepare template data
    template_data = {
        'request': request,
        'colors': KEARNEY_COLORS,
        'current_price': f"${current_price:,.2f}",
        'forecast_price': f"${forecast_price:,.2f}",
        'price_change_pct': f"{price_change_pct:+.1f}%",
        'confidence_score': f"{consensus['confidence_score']*100:.0f}%",
        'signal': consensus['consensus_signal'].upper(),
        'signal_class': 'text-purple-400' if 'buy' in consensus['consensus_signal'] else 'text-gray-400',
        'chart_data': {
            'dates': dates,
            'actual_prices': actual_prices,
            'predicted_prices': predicted_prices[-len(actual_prices):] if len(predicted_prices) >= len(actual_prices) else predicted_prices
        },
        'performance': {
            'mape': f"{backtest['mape']:.2f}%",
            'cumulative_return': f"{backtest['cumulative_return_pct']:+.1f}%",
            'sharpe_ratio': f"{backtest['sharpe_ratio']:.2f}",
            'avg_confidence': f"{backtest['avg_confidence']*100:.0f}%"
        },
        'signals': {
            'total': signal_summary['total_signals'],
            'bullish': signal_summary['bullish_signals'],
            'bearish': signal_summary['bearish_signals'],
            'quality': signal_summary['signal_quality'].upper()
        },
        'recommendations': recommendations,
        'role_contributions': role_contributions,
        'market_context': {
            'trend_5d': f"{context['trend_5d']:+.1f}%",
            'trend_20d': f"{context['trend_20d']:+.1f}%",
            'volatility': f"{context['volatility_20d']:.1f}%",
            'price_level': f"{context['price_percentile']:.0f}th percentile"
        },
        'last_update': _data_cache['last_update'].strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return templates.TemplateResponse("dashboard_zen_consensus.html", template_data)

@app.get("/api/refresh")
async def refresh_predictions():
    """Refresh predictions"""
    update_predictions()
    return {"status": "success", "timestamp": _data_cache['last_update'].isoformat()}

@app.get("/api/signals")
async def get_signals():
    """Get latest signals"""
    return _data_cache['signals']

@app.get("/api/consensus")
async def get_consensus():
    """Get latest consensus"""
    return _data_cache['consensus_result']

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)