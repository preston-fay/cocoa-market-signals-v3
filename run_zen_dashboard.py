#!/usr/bin/env python3
"""
Run Zen Consensus Dashboard - Standalone Script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import json
from datetime import datetime
import uvicorn
import logging

# Simple in-memory data
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Kearney Colors
COLORS = {
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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Simple dashboard with mock data"""
    
    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
    base_price = 7573
    
    # Actual prices with some volatility
    actual_prices = []
    price = base_price
    for i in range(90):
        price = price * (1 + np.random.normal(0, 0.02))
        price = max(3282, min(12565, price))
        actual_prices.append(price)
    
    # Predicted prices (slightly lagged)
    predicted_prices = [actual_prices[0]]
    for i in range(1, 90):
        # Predictions lag reality slightly
        predicted = actual_prices[i-1] * 0.7 + predicted_prices[-1] * 0.3
        predicted_prices.append(predicted)
    
    # Current values
    current_price = actual_prices[-1]
    forecast_price = current_price * 1.012  # 1.2% increase forecast
    
    template_data = {
        'request': request,
        'colors': COLORS,
        'current_price': f"${current_price:,.2f}",
        'forecast_price': f"${forecast_price:,.2f}",
        'price_change_pct': f"+1.2%",
        'confidence_score': "85%",
        'signal': "BUY",
        'signal_class': 'text-purple-400',
        'chart_data': {
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'actual_prices': actual_prices,
            'predicted_prices': predicted_prices
        },
        'performance': {
            'mape': "3.5%",
            'cumulative_return': "+12.3%",
            'sharpe_ratio': "1.45",
            'avg_confidence': "82%"
        },
        'signals': {
            'total': 8,
            'bullish': 5,
            'bearish': 2,
            'quality': 'HIGH'
        },
        'recommendations': [
            {'icon': '🚀', 'text': 'Strong BUY signal - Target: $8,541'},
            {'icon': '📊', 'text': 'All model roles agree on bullish outlook'},
            {'icon': '📈', 'text': 'Volume surge detected in recent trading'}
        ],
        'role_contributions': [
            {
                'role': 'Neutral Analyst',
                'prediction': '$8,520',
                'change_pct': '+1.1%',
                'confidence': '78%'
            },
            {
                'role': 'Supportive Trader', 
                'prediction': '$8,580',
                'change_pct': '+1.8%',
                'confidence': '92%'
            },
            {
                'role': 'Critical Risk Manager',
                'prediction': '$8,490',
                'change_pct': '+0.7%',
                'confidence': '85%'
            }
        ],
        'market_context': {
            'trend_5d': '+2.3%',
            'trend_20d': '+5.1%',
            'volatility': '42.5%',
            'price_level': '73rd percentile'
        },
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return templates.TemplateResponse("dashboard_zen_consensus.html", template_data)

@app.get("/api/refresh")
async def refresh():
    return {"status": "success", "timestamp": datetime.now().isoformat()}

@app.get("/api/consensus")
async def consensus():
    return {
        "consensus": {
            "current_price": 8440.0,
            "consensus_forecast": 8541.0,
            "consensus_signal": "buy",
            "confidence_score": 0.85
        }
    }

@app.get("/api/signals") 
async def signals():
    return {
        "summary": {
            "total_signals": 8,
            "bullish_signals": 5,
            "bearish_signals": 2,
            "signal_quality": "high"
        }
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING ZEN CONSENSUS DASHBOARD")
    print("="*60)
    print("\nOpen your browser to: http://localhost:8000")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)