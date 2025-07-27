#!/usr/bin/env python3
"""
Historical Analysis Dashboard - Shows REAL backtested performance
Uses ACTUAL predictions from database - NO FAKE DATA
"""
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from sqlmodel import Session, select, and_, func
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import json
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.database import engine
from app.models.prediction import Prediction
from app.models.signal import Signal
from app.models.price_data import PriceData

app = FastAPI(title="Historical Backtesting Dashboard")

def get_historical_performance():
    """Get REAL historical performance from database"""
    with Session(engine) as session:
        # Get all predictions with outcomes
        all_predictions = session.exec(
            select(Prediction)
            .where(Prediction.actual_price.is_not(None))
            .order_by(Prediction.created_at)
        ).all()
        
        # Group by horizon
        by_horizon = {}
        for pred in all_predictions:
            h = pred.prediction_horizon
            if h not in by_horizon:
                by_horizon[h] = {
                    'predictions': [],
                    'mape': 0,
                    'directional': 0,
                    'count': 0
                }
            
            by_horizon[h]['predictions'].append({
                'date': pred.created_at.strftime('%Y-%m-%d'),
                'target_date': pred.target_date.strftime('%Y-%m-%d'),
                'predicted': float(pred.predicted_price),
                'actual': float(pred.actual_price),
                'error_pct': float(pred.error_percentage),
                'direction_correct': (pred.predicted_price > pred.current_price) == (pred.actual_price > pred.current_price)
            })
        
        # Calculate real metrics
        for horizon, data in by_horizon.items():
            if data['predictions']:
                data['mape'] = np.mean([p['error_pct'] for p in data['predictions']])
                data['directional'] = np.mean([p['direction_correct'] for p in data['predictions']]) * 100
                data['count'] = len(data['predictions'])
        
        return by_horizon

def get_price_history(start_date: date, end_date: date):
    """Get price history for period"""
    with Session(engine) as session:
        prices = session.exec(
            select(PriceData)
            .where(and_(
                PriceData.source == "Yahoo Finance",
                PriceData.date >= start_date,
                PriceData.date <= end_date
            ))
            .order_by(PriceData.date)
        ).all()
        
        return [{
            'date': p.date.strftime('%Y-%m-%d'),
            'price': float(p.price)
        } for p in prices]

def get_signals_history():
    """Get all historical signals"""
    with Session(engine) as session:
        signals = session.exec(
            select(Signal)
            .where(Signal.source.in_(["zen_consensus", "zen_consensus_historical"]))
            .order_by(Signal.signal_date)
        ).all()
        
        return [{
            'date': s.signal_date.strftime('%Y-%m-%d'),
            'type': s.signal_name,
            'direction': s.signal_direction,
            'strength': float(s.signal_strength),
            'confidence': float(s.confidence)
        } for s in signals]

@app.get("/", response_class=HTMLResponse)
async def historical_dashboard():
    """Main historical analysis dashboard"""
    
    # Get real performance data
    performance = get_historical_performance()
    
    # Get full price history
    with Session(engine) as session:
        date_range = session.exec(
            select(func.min(PriceData.date), func.max(PriceData.date))
            .where(PriceData.source == "Yahoo Finance")
        ).first()
    
    start_date, end_date = date_range
    prices = get_price_history(start_date, end_date)
    signals = get_signals_history()
    
    # Calculate overall metrics
    all_predictions = []
    for h_data in performance.values():
        all_predictions.extend(h_data['predictions'])
    
    if all_predictions:
        overall_mape = np.mean([p['error_pct'] for p in all_predictions])
        overall_directional = np.mean([p['direction_correct'] for p in all_predictions]) * 100
    else:
        overall_mape = 0
        overall_directional = 0
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Historical Backtesting Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            background-color: #000000;
            color: #e9ecef;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            line-height: 1.6;
        }}
        
        .header {{
            background: #272b30;
            padding: 2rem 0;
            text-align: center;
            border-bottom: 3px solid #6f42c1;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            color: #e9ecef;
            font-weight: 300;
            letter-spacing: 2px;
        }}
        
        .header p {{
            color: #999999;
            margin-top: 0.5rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}
        
        .metric-card {{
            background: #272b30;
            border: 1px solid #7a8288;
            padding: 1.5rem;
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 300;
            color: #6f42c1;
            margin: 0.5rem 0;
        }}
        
        .metric-label {{
            color: #999999;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .chart-section {{
            background: #272b30;
            border: 1px solid #7a8288;
            padding: 2rem;
            margin-bottom: 2rem;
        }}
        
        .chart-title {{
            font-size: 1.5rem;
            color: #e9ecef;
            margin-bottom: 1.5rem;
            font-weight: 300;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
        }}
        
        .horizon-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 2rem;
        }}
        
        .horizon-table th {{
            background: #52575c;
            color: #e9ecef;
            padding: 1rem;
            text-align: left;
            font-weight: 400;
            text-transform: uppercase;
            font-size: 0.875rem;
            letter-spacing: 1px;
        }}
        
        .horizon-table td {{
            padding: 1rem;
            border-bottom: 1px solid #7a8288;
            color: #e9ecef;
        }}
        
        .horizon-table tr:hover {{
            background: #52575c;
        }}
        
        .good {{ color: #6f42c1; }}
        .poor {{ color: #999999; }}
        
        .timeline-controls {{
            background: #272b30;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid #7a8288;
        }}
        
        .date-range {{
            display: flex;
            align-items: center;
            gap: 1rem;
            justify-content: center;
        }}
        
        .date-input {{
            background: #000000;
            border: 1px solid #7a8288;
            color: #e9ecef;
            padding: 0.5rem 1rem;
            font-size: 1rem;
        }}
        
        .btn {{
            background: #6f42c1;
            color: #e9ecef;
            border: none;
            padding: 0.5rem 1.5rem;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.2s;
        }}
        
        .btn:hover {{
            background: #52575c;
        }}
        
        .signal-marker {{
            position: absolute;
            width: 2px;
            height: 100%;
            top: 0;
        }}
        
        .signal-buy {{ background: #6f42c1; }}
        .signal-sell {{ background: #999999; }}
        
        .footer {{
            text-align: center;
            padding: 3rem 0;
            color: #999999;
            font-size: 0.875rem;
            border-top: 1px solid #7a8288;
            margin-top: 4rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>HISTORICAL BACKTESTING ANALYSIS</h1>
        <p>Real Model Performance • {len(all_predictions)} Predictions Evaluated</p>
    </div>
    
    <div class="container">
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Overall MAPE</div>
                <div class="metric-value">{overall_mape:.1f}%</div>
                <div class="metric-label">Mean Absolute Percentage Error</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Directional Accuracy</div>
                <div class="metric-value">{overall_directional:.1f}%</div>
                <div class="metric-label">Correct Direction Predictions</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Total Predictions</div>
                <div class="metric-value">{len(all_predictions)}</div>
                <div class="metric-label">With Actual Outcomes</div>
            </div>
        </div>
        
        <div class="chart-section">
            <h2 class="chart-title">Historical Price with Predictions</h2>
            <div class="timeline-controls">
                <div class="date-range">
                    <input type="date" id="startDate" class="date-input" value="{start_date}">
                    <span>to</span>
                    <input type="date" id="endDate" class="date-input" value="{end_date}">
                    <button class="btn" onclick="updateChart()">Update</button>
                </div>
            </div>
            <div class="chart-container">
                <canvas id="priceChart"></canvas>
            </div>
        </div>
        
        <div class="chart-section">
            <h2 class="chart-title">Performance by Prediction Horizon</h2>
            <table class="horizon-table">
                <thead>
                    <tr>
                        <th>Horizon</th>
                        <th>Predictions</th>
                        <th>MAPE</th>
                        <th>Directional Accuracy</th>
                        <th>Best Error</th>
                        <th>Worst Error</th>
                    </tr>
                </thead>
                <tbody>
                    {generate_horizon_rows(performance)}
                </tbody>
            </table>
        </div>
        
        <div class="chart-section">
            <h2 class="chart-title">Error Distribution by Horizon</h2>
            <div class="chart-container" style="height: 300px;">
                <canvas id="errorChart"></canvas>
            </div>
        </div>
        
        <div class="chart-section">
            <h2 class="chart-title">Trading Signals History</h2>
            <div class="chart-container">
                <canvas id="signalChart"></canvas>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Historical Backtesting Dashboard • Real Data Only • No Synthetic Results</p>
        <p>Following Preston Dev Setup Standards</p>
    </div>
    
    <script>
        // Real data from database
        const prices = {json.dumps(prices)};
        const signals = {json.dumps(signals)};
        const performance = {json.dumps(performance)};
        
        // Chart defaults
        Chart.defaults.color = '#999999';
        Chart.defaults.borderColor = '#52575c';
        
        // Price chart with predictions
        const priceCtx = document.getElementById('priceChart').getContext('2d');
        let priceChart;
        
        function initPriceChart(startDate, endDate) {{
            // Filter data by date range
            const filteredPrices = prices.filter(p => 
                p.date >= startDate && p.date <= endDate
            );
            
            // Prepare prediction overlays
            const predictions = [];
            for (const [horizon, data] of Object.entries(performance)) {{
                data.predictions.forEach(pred => {{
                    if (pred.date >= startDate && pred.date <= endDate) {{
                        predictions.push(pred);
                    }}
                }});
            }}
            
            const datasets = [{{
                label: 'Actual Price',
                data: filteredPrices.map(p => ({{x: p.date, y: p.price}})),
                borderColor: '#6f42c1',
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1
            }}];
            
            // Add prediction points
            if (predictions.length > 0) {{
                datasets.push({{
                    label: 'Predictions',
                    data: predictions.map(p => ({{x: p.target_date, y: p.predicted}})),
                    borderColor: '#e9ecef',
                    backgroundColor: '#e9ecef',
                    borderWidth: 0,
                    pointRadius: 3,
                    showLine: false
                }});
            }}
            
            if (priceChart) {{
                priceChart.destroy();
            }}
            
            priceChart = new Chart(priceCtx, {{
                type: 'line',
                data: {{ datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{
                        mode: 'index',
                        intersect: false
                    }},
                    plugins: {{
                        legend: {{
                            labels: {{ color: '#e9ecef' }}
                        }},
                        tooltip: {{
                            backgroundColor: '#000000',
                            titleColor: '#e9ecef',
                            bodyColor: '#999999',
                            borderColor: '#7a8288',
                            borderWidth: 1,
                            padding: 12
                        }}
                    }},
                    scales: {{
                        x: {{
                            type: 'time',
                            time: {{
                                unit: 'month'
                            }},
                            ticks: {{ color: '#999999' }},
                            grid: {{ display: false }}
                        }},
                        y: {{
                            ticks: {{
                                color: '#999999',
                                callback: function(value) {{
                                    return '$' + value.toLocaleString();
                                }}
                            }},
                            grid: {{ display: false }}
                        }}
                    }}
                }}
            }});
        }}
        
        // Error distribution chart
        const errorCtx = document.getElementById('errorChart').getContext('2d');
        const horizonLabels = Object.keys(performance).map(h => h + ' days');
        const mapeData = Object.values(performance).map(d => d.mape || 0);
        const directionalData = Object.values(performance).map(d => d.directional || 0);
        
        new Chart(errorCtx, {{
            type: 'bar',
            data: {{
                labels: horizonLabels,
                datasets: [
                    {{
                        label: 'MAPE (%)',
                        data: mapeData,
                        backgroundColor: '#6f42c1',
                        borderColor: '#6f42c1',
                        borderWidth: 0
                    }},
                    {{
                        label: 'Directional Accuracy (%)',
                        data: directionalData,
                        backgroundColor: '#999999',
                        borderColor: '#999999',
                        borderWidth: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#e9ecef' }}
                    }},
                    tooltip: {{
                        backgroundColor: '#000000',
                        titleColor: '#e9ecef',
                        bodyColor: '#999999',
                        borderColor: '#7a8288',
                        borderWidth: 1,
                        padding: 12
                    }}
                }},
                scales: {{
                    x: {{
                        ticks: {{ color: '#999999' }},
                        grid: {{ display: false }}
                    }},
                    y: {{
                        ticks: {{
                            color: '#999999',
                            callback: function(value) {{
                                return value + '%';
                            }}
                        }},
                        grid: {{ display: false }}
                    }}
                }}
            }}
        }});
        
        // Initialize price chart
        initPriceChart('{start_date}', '{end_date}');
        
        // Update chart function
        function updateChart() {{
            const startDate = document.getElementById('startDate').value;
            const endDate = document.getElementById('endDate').value;
            initPriceChart(startDate, endDate);
        }}
        
        // Signal chart
        const signalCtx = document.getElementById('signalChart').getContext('2d');
        const signalData = signals.map(s => ({{
            x: s.date,
            y: s.strength,
            direction: s.direction,
            confidence: s.confidence
        }}));
        
        new Chart(signalCtx, {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Buy Signals',
                    data: signalData.filter(s => s.direction === 'bullish'),
                    backgroundColor: '#6f42c1',
                    borderColor: '#6f42c1',
                    pointRadius: 6
                }},
                {{
                    label: 'Sell Signals',
                    data: signalData.filter(s => s.direction === 'bearish'),
                    backgroundColor: '#999999',
                    borderColor: '#999999',
                    pointRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#e9ecef' }}
                    }},
                    tooltip: {{
                        backgroundColor: '#000000',
                        titleColor: '#e9ecef',
                        bodyColor: '#999999',
                        borderColor: '#7a8288',
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {{
                            label: function(context) {{
                                return [
                                    context.dataset.label,
                                    'Strength: ' + context.parsed.y.toFixed(1),
                                    'Confidence: ' + (context.raw.confidence * 100).toFixed(0) + '%'
                                ];
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        type: 'time',
                        time: {{
                            unit: 'month'
                        }},
                        ticks: {{ color: '#999999' }},
                        grid: {{ display: false }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Signal Strength',
                            color: '#999999'
                        }},
                        ticks: {{ color: '#999999' }},
                        grid: {{ display: false }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    return HTMLResponse(content=html)

def generate_horizon_rows(performance):
    """Generate table rows for horizon performance"""
    if not performance:
        return "<tr><td colspan='6' style='text-align: center; color: #999999;'>No performance data available</td></tr>"
    
    rows = ""
    for horizon in sorted(performance.keys()):
        data = performance[horizon]
        if data['predictions']:
            errors = [p['error_pct'] for p in data['predictions']]
            best_error = min(errors)
            worst_error = max(errors)
            
            mape_class = "good" if data['mape'] < 5 else "poor"
            dir_class = "good" if data['directional'] > 60 else "poor"
            
            rows += f"""
            <tr>
                <td>{horizon} days</td>
                <td>{data['count']}</td>
                <td class="{mape_class}">{data['mape']:.1f}%</td>
                <td class="{dir_class}">{data['directional']:.1f}%</td>
                <td class="good">{best_error:.1f}%</td>
                <td class="poor">{worst_error:.1f}%</td>
            </tr>
            """
    
    return rows

if __name__ == "__main__":
    import uvicorn
    print("\nStarting Historical Analysis Dashboard")
    print("Visit: http://localhost:8006")
    print("This shows REAL backtested performance - no fake data!\n")
    uvicorn.run(app, host="127.0.0.1", port=8006)