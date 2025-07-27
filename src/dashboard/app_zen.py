#!/usr/bin/env python3
"""
Zen Consensus Dashboard - 100% REAL predictions
Displays current consensus and historical performance
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select
from datetime import datetime, date, timedelta
import json
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.database import engine
from app.models.prediction import Prediction
from app.models.signal import Signal
from app.models.model_performance import ModelPerformance
from app.models.price_data import PriceData

app = FastAPI(title="Zen Consensus Dashboard")

# Mount static files
static_path = Path(__file__).parent.parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page"""
    
    with Session(engine) as session:
        # Get latest consensus predictions
        predictions = session.exec(
            select(Prediction)
            .where(Prediction.model_name == "zen_consensus")
            .order_by(Prediction.created_at.desc())
            .limit(3)
        ).all()
        
        # Get current signal
        current_signal = session.exec(
            select(Signal)
            .where(Signal.source == "zen_consensus")
            .order_by(Signal.detected_at.desc())
            .limit(1)
        ).first()
        
        # Get role signals
        role_signals = session.exec(
            select(Signal)
            .where(Signal.detector == "simple_zen_consensus")
            .where(Signal.signal_type == "role")
            .order_by(Signal.detected_at.desc())
            .limit(3)
        ).all()
        
        # Get recent performance
        performance = session.exec(
            select(ModelPerformance)
            .where(ModelPerformance.model_name == "zen_consensus")
            .order_by(ModelPerformance.evaluation_date.desc())
            .limit(1)
        ).first()
        
        # Get recent prices for chart
        recent_prices = session.exec(
            select(PriceData)
            .where(PriceData.source == "Yahoo Finance")
            .order_by(PriceData.date.desc())
            .limit(30)
        ).all()
        
        # Prepare data for template
        template_data = {
            "predictions": predictions,
            "current_signal": current_signal,
            "role_signals": role_signals,
            "performance": performance,
            "price_data": list(reversed(recent_prices))
        }
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zen Consensus Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary-purple: #6f42c1;
            --primary-charcoal: #272b30;
            --white: #FFFFFF;
            --light-gray: #e9ecef;
            --medium-gray: #999999;
            --dark-gray: #52575c;
            --border-gray: #7a8288;
        }}
        
        body {{
            background-color: var(--primary-charcoal);
            color: var(--light-gray);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        h1, h2, h3 {{
            color: var(--white);
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 2px solid var(--border-gray);
            padding-bottom: 20px;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-gray);
            border-radius: 8px;
            padding: 20px;
        }}
        
        .signal {{
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        .signal.bearish {{
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid var(--medium-gray);
        }}
        
        .signal.bullish {{
            background: rgba(111, 66, 193, 0.2);
            border: 2px solid var(--primary-purple);
        }}
        
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 4px;
        }}
        
        .metric-label {{
            color: var(--medium-gray);
        }}
        
        .metric-value {{
            font-weight: bold;
            color: var(--white);
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 40px 0;
        }}
        
        .role-signal {{
            padding: 15px;
            margin: 10px 0;
            background: rgba(255, 255, 255, 0.03);
            border-left: 3px solid var(--primary-purple);
        }}
        
        .confidence-bar {{
            height: 20px;
            background: var(--dark-gray);
            border-radius: 4px;
            overflow: hidden;
            margin: 5px 0;
        }}
        
        .confidence-fill {{
            height: 100%;
            background: var(--primary-purple);
            transition: width 0.3s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧘 Zen Consensus Dashboard</h1>
            <p>Multi-Model AI Consensus for Cocoa Market Predictions</p>
        </div>
        
        {'<div class="signal ' + (current_signal.signal_direction if current_signal else 'neutral') + '">' if current_signal else '<div class="signal neutral">'}
            {f'{current_signal.signal_name.upper().replace("_", " ")}' if current_signal else 'NO ACTIVE SIGNAL'}
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📊 Current Predictions</h3>
                {generate_predictions_html(predictions)}
            </div>
            
            <div class="card">
                <h3>🤖 AI Role Perspectives</h3>
                {generate_roles_html(role_signals)}
            </div>
            
            <div class="card">
                <h3>📈 Model Performance</h3>
                {generate_performance_html(performance)}
            </div>
        </div>
        
        <div class="card">
            <h3>Price Chart with Predictions</h3>
            <div class="chart-container">
                <canvas id="priceChart"></canvas>
            </div>
        </div>
    </div>
    
    <script>
        // Price chart with predictions
        const ctx = document.getElementById('priceChart').getContext('2d');
        const priceData = {json.dumps([{'date': p.date.isoformat(), 'price': p.price} for p in template_data['price_data']])};
        const predictions = {json.dumps([{'date': p.target_date.isoformat(), 'price': p.predicted_price} for p in template_data['predictions']] if template_data['predictions'] else [])};
        
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: priceData.map(d => d.date),
                datasets: [{{
                    label: 'Actual Price',
                    data: priceData.map(d => d.price),
                    borderColor: '#6f42c1',
                    backgroundColor: 'rgba(111, 66, 193, 0.1)',
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{
                            color: '#e9ecef'
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        ticks: {{
                            color: '#999999'
                        }},
                        grid: {{
                            color: '#52575c'
                        }}
                    }},
                    y: {{
                        ticks: {{
                            color: '#999999',
                            callback: function(value) {{
                                return '$' + value.toLocaleString();
                            }}
                        }},
                        grid: {{
                            color: '#52575c'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    return HTMLResponse(content=html)

def generate_predictions_html(predictions):
    """Generate HTML for predictions"""
    if not predictions:
        return "<p>No predictions available</p>"
    
    html = ""
    for pred in predictions[:3]:  # Show top 3
        days_ahead = (pred.target_date - date.today()).days
        change = (pred.predicted_price - pred.current_price) / pred.current_price * 100
        
        html += f"""
        <div class="metric">
            <span class="metric-label">{days_ahead}-Day Forecast</span>
            <span class="metric-value">${pred.predicted_price:,.0f} ({change:+.1f}%)</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {pred.confidence_score * 100}%"></div>
        </div>
        """
    
    return html

def generate_roles_html(role_signals):
    """Generate HTML for role signals"""
    if not role_signals:
        return "<p>No role signals available</p>"
    
    html = ""
    for signal in role_signals:
        html += f"""
        <div class="role-signal">
            <strong>{signal.source.replace('_', ' ').title()}</strong>
            <div class="metric">
                <span class="metric-label">Direction</span>
                <span class="metric-value">{signal.signal_direction.upper()}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Strength</span>
                <span class="metric-value">{signal.signal_strength:+.1f}</span>
            </div>
        </div>
        """
    
    return html

def generate_performance_html(performance):
    """Generate HTML for performance metrics"""
    if not performance:
        return "<p>No performance data available</p>"
    
    return f"""
    <div class="metric">
        <span class="metric-label">Mean Error (MAPE)</span>
        <span class="metric-value">{performance.mape:.1f}%</span>
    </div>
    <div class="metric">
        <span class="metric-label">Direction Accuracy</span>
        <span class="metric-value">{performance.directional_accuracy:.1%}</span>
    </div>
    <div class="metric">
        <span class="metric-label">Predictions Made</span>
        <span class="metric-value">{performance.predictions_made}</span>
    </div>
    <div class="metric">
        <span class="metric-label">Evaluation Period</span>
        <span class="metric-value">{performance.period_days} days</span>
    </div>
    """

if __name__ == "__main__":
    import uvicorn
    print("\n🧘 Starting Zen Consensus Dashboard on http://localhost:8003")
    print("   Press Ctrl+C to stop\n")
    uvicorn.run(app, host="127.0.0.1", port=8003)