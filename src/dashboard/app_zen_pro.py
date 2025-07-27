#!/usr/bin/env python3
"""
Zen Consensus Professional Dashboard
Meets ALL user requirements with actual vs predicted, month navigation, KPIs
"""
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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
from app.models.model_performance import ModelPerformance
from app.models.price_data import PriceData

app = FastAPI(title="Zen Consensus Professional Dashboard")

# Mount static files if available
static_path = Path(__file__).parent.parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

def get_predictions_with_actuals(start_date: date, end_date: date):
    """Get predictions with their actual outcomes"""
    with Session(engine) as session:
        # Get predictions that have target dates in the range
        predictions = session.exec(
            select(Prediction)
            .where(and_(
                Prediction.model_name == "zen_consensus",
                Prediction.target_date >= start_date,
                Prediction.target_date <= end_date
            ))
            .order_by(Prediction.target_date)
        ).all()
        
        # Get actual prices for comparison
        actuals = session.exec(
            select(PriceData)
            .where(and_(
                PriceData.source == "Yahoo Finance",
                PriceData.date >= start_date,
                PriceData.date <= end_date
            ))
            .order_by(PriceData.date)
        ).all()
        
        # Create lookup for actual prices
        actual_prices = {p.date: p.price for p in actuals}
        
        # Update predictions with actuals
        results = []
        for pred in predictions:
            actual = actual_prices.get(pred.target_date)
            if actual:
                error = actual - pred.predicted_price
                error_pct = abs(error) / actual * 100
                direction_correct = (pred.predicted_price > pred.current_price) == (actual > pred.current_price)
            else:
                error = None
                error_pct = None
                direction_correct = None
            
            results.append({
                'target_date': pred.target_date.isoformat() if pred.target_date else None,
                'prediction_date': pred.created_at.date().isoformat() if pred.created_at else None,
                'horizon': pred.prediction_horizon,
                'predicted': pred.predicted_price,
                'actual': actual,
                'current': pred.current_price,
                'confidence': pred.confidence_score,
                'error': error,
                'error_pct': error_pct,
                'direction_correct': direction_correct
            })
        
        return results, list(actuals)

def calculate_kpis(predictions_data):
    """Calculate KPIs from predictions data"""
    evaluated = [p for p in predictions_data if p['actual'] is not None]
    
    if not evaluated:
        return {
            'overall_accuracy': None,
            'directional_accuracy': None,
            'avg_confidence': None,
            'predictions_made': len(predictions_data),
            'predictions_evaluated': 0
        }
    
    mape = np.mean([p['error_pct'] for p in evaluated])
    directional = np.mean([p['direction_correct'] for p in evaluated if p['direction_correct'] is not None])
    avg_confidence = np.mean([p['confidence'] for p in predictions_data])
    
    # Calculate by horizon
    horizon_accuracy = {}
    for horizon in [1, 7, 30]:
        horizon_preds = [p for p in evaluated if p['horizon'] == horizon]
        if horizon_preds:
            horizon_accuracy[f"{horizon}d"] = {
                'mape': np.mean([p['error_pct'] for p in horizon_preds]),
                'count': len(horizon_preds)
            }
    
    return {
        'overall_accuracy': 100 - mape,  # Convert MAPE to accuracy
        'mape': mape,
        'directional_accuracy': directional * 100,
        'avg_confidence': avg_confidence * 100,
        'predictions_made': len(predictions_data),
        'predictions_evaluated': len(evaluated),
        'horizon_accuracy': horizon_accuracy
    }

def get_model_performance_history():
    """Get historical model performance"""
    with Session(engine) as session:
        performance = session.exec(
            select(ModelPerformance)
            .where(ModelPerformance.model_name == "zen_consensus")
            .order_by(ModelPerformance.evaluation_date.desc())
            .limit(12)  # Last 12 evaluations
        ).all()
        
        return [{
            'date': p.evaluation_date.isoformat() if p.evaluation_date else None,
            'mape': p.mape,
            'directional_accuracy': p.directional_accuracy,
            'predictions_evaluated': p.predictions_evaluated
        } for p in performance]

def get_signals_with_outcomes(start_date: date, end_date: date):
    """Get signals and their outcomes"""
    with Session(engine) as session:
        signals = session.exec(
            select(Signal)
            .where(and_(
                Signal.source == "zen_consensus",
                Signal.signal_date >= datetime.combine(start_date, datetime.min.time()),
                Signal.signal_date <= datetime.combine(end_date, datetime.max.time())
            ))
            .order_by(Signal.signal_date.desc())
        ).all()
        
        return [{
            'date': s.signal_date.date().isoformat() if s.signal_date else None,
            'signal': s.signal_name,
            'direction': s.signal_direction,
            'strength': s.signal_strength,
            'confidence': s.confidence,
            'description': s.description
        } for s in signals]

@app.get("/", response_class=HTMLResponse)
async def dashboard(month: str = Query(default=None, description="Month in YYYY-MM format")):
    """Main dashboard with month navigation"""
    
    # Determine the month to display
    if month:
        try:
            selected_date = datetime.strptime(month, "%Y-%m").date()
        except:
            selected_date = date.today().replace(day=1)
    else:
        selected_date = date.today().replace(day=1)
    
    # Calculate date range for the month
    start_date = selected_date
    end_date = (selected_date + relativedelta(months=1) - timedelta(days=1))
    
    # Get data
    predictions_data, price_data = get_predictions_with_actuals(start_date, end_date)
    kpis = calculate_kpis(predictions_data)
    signals = get_signals_with_outcomes(start_date, end_date)
    performance_history = get_model_performance_history()
    
    # Generate month navigation
    current_month = selected_date
    prev_month = (current_month - relativedelta(months=1)).strftime("%Y-%m")
    next_month = (current_month + relativedelta(months=1)).strftime("%Y-%m")
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zen Consensus Dashboard - {selected_date.strftime('%B %Y')}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
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
        
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            background-color: var(--primary-charcoal);
            color: var(--light-gray);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            margin: 0;
            padding: 0;
        }}
        
        .header {{
            background: linear-gradient(135deg, var(--primary-purple) 0%, var(--dark-gray) 100%);
            padding: 20px 0;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        
        .header h1 {{
            margin: 0;
            color: var(--white);
            font-size: 2.5em;
        }}
        
        .month-navigation {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin: 20px 0;
        }}
        
        .nav-button {{
            background: var(--primary-purple);
            color: var(--white);
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.3s;
        }}
        
        .nav-button:hover {{
            background: var(--dark-gray);
            transform: translateY(-1px);
        }}
        
        .current-month {{
            font-size: 1.5em;
            font-weight: bold;
            color: var(--white);
            min-width: 200px;
            text-align: center;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .kpi-card {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-gray);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s;
        }}
        
        .kpi-card:hover {{
            background: rgba(255, 255, 255, 0.08);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(111, 66, 193, 0.2);
        }}
        
        .kpi-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: var(--primary-purple);
            margin: 10px 0;
        }}
        
        .kpi-label {{
            color: var(--medium-gray);
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .kpi-trend {{
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .trend-up {{
            color: var(--primary-purple);
        }}
        
        .trend-down {{
            color: var(--medium-gray);
        }}
        
        .chart-container {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-gray);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        
        .chart-title {{
            color: var(--white);
            margin-bottom: 20px;
            font-size: 1.3em;
        }}
        
        .section-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        @media (max-width: 768px) {{
            .section-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        
        .data-table th {{
            background: var(--dark-gray);
            color: var(--white);
            padding: 10px;
            text-align: left;
            font-weight: 600;
        }}
        
        .data-table td {{
            padding: 10px;
            border-bottom: 1px solid var(--border-gray);
        }}
        
        .data-table tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        
        .accuracy-good {{
            color: var(--primary-purple);
        }}
        
        .accuracy-poor {{
            color: var(--medium-gray);
        }}
        
        .signal-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        
        .signal-bullish {{
            background: rgba(111, 66, 193, 0.2);
            color: var(--primary-purple);
            border: 1px solid var(--primary-purple);
        }}
        
        .signal-bearish {{
            background: rgba(255, 255, 255, 0.1);
            color: var(--medium-gray);
            border: 1px solid var(--medium-gray);
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: var(--medium-gray);
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧘 Zen Consensus Dashboard</h1>
        <p style="color: var(--light-gray); margin: 5px 0;">AI-Powered Cocoa Market Predictions</p>
    </div>
    
    <div class="container">
        <!-- Month Navigation -->
        <div class="month-navigation">
            <a href="/?month={prev_month}" class="nav-button">← Previous</a>
            <div class="current-month">{selected_date.strftime('%B %Y')}</div>
            <a href="/?month={next_month}" class="nav-button">Next →</a>
        </div>
        
        <!-- KPI Dashboard -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Overall Accuracy</div>
                <div class="kpi-value">{f"{kpis['overall_accuracy']:.1f}%" if kpis.get('overall_accuracy') else "—"}</div>
                <div class="kpi-trend {' trend-up' if kpis.get('overall_accuracy') and kpis['overall_accuracy'] > 90 else 'trend-down'}">
                    {f"MAPE: {kpis['mape']:.1f}%" if kpis.get('mape') else "No data"}
                </div>
            </div>
            
            <div class="kpi-card">
                <div class="kpi-label">Directional Accuracy</div>
                <div class="kpi-value">{f"{kpis['directional_accuracy']:.1f}%" if kpis.get('directional_accuracy') else "—"}</div>
                <div class="kpi-trend">Predicted direction correctly</div>
            </div>
            
            <div class="kpi-card">
                <div class="kpi-label">Predictions Made</div>
                <div class="kpi-value">{kpis.get('predictions_made', 0)}</div>
                <div class="kpi-trend">{kpis.get('predictions_evaluated', 0)} evaluated</div>
            </div>
            
            <div class="kpi-card">
                <div class="kpi-label">Avg Confidence</div>
                <div class="kpi-value">{f"{kpis['avg_confidence']:.1f}%" if kpis.get('avg_confidence') else "—"}</div>
                <div class="kpi-trend">Model confidence score</div>
            </div>
        </div>
        
        <!-- Actual vs Predicted Chart -->
        <div class="chart-container">
            <h3 class="chart-title">Actual Prices vs Predictions</h3>
            <div style="height: 400px;">
                <canvas id="actualVsPredictedChart"></canvas>
            </div>
        </div>
        
        <div class="section-grid">
            <!-- Prediction Details Table -->
            <div class="chart-container">
                <h3 class="chart-title">Prediction Details</h3>
                <div style="max-height: 400px; overflow-y: auto;">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Predicted</th>
                                <th>Actual</th>
                                <th>Error</th>
                                <th>Horizon</th>
                            </tr>
                        </thead>
                        <tbody>
                            {generate_prediction_rows(predictions_data[:10])}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Signals Generated -->
            <div class="chart-container">
                <h3 class="chart-title">Signals Generated</h3>
                <div style="max-height: 400px; overflow-y: auto;">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Signal</th>
                                <th>Strength</th>
                                <th>Confidence</th>
                            </tr>
                        </thead>
                        <tbody>
                            {generate_signal_rows(signals[:10])}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- Horizon Accuracy Analysis -->
        <div class="chart-container">
            <h3 class="chart-title">Prediction Accuracy by Horizon</h3>
            <div style="height: 300px;">
                <canvas id="horizonAccuracyChart"></canvas>
            </div>
        </div>
        
        <!-- Performance Trend -->
        <div class="chart-container">
            <h3 class="chart-title">Model Performance Trend</h3>
            <div style="height: 300px;">
                <canvas id="performanceTrendChart"></canvas>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Zen Consensus Dashboard • 100% Real Data • Updated Daily</p>
    </div>
    
    <script>
        // Prepare data for charts
        const predictions = {json.dumps(predictions_data)};
        const priceData = {json.dumps([{'date': p.date.isoformat(), 'price': p.price} for p in price_data])};
        const performanceHistory = {json.dumps(performance_history)};
        const horizonData = {json.dumps(kpis.get('horizon_accuracy', {}))};
        
        // Chart configuration
        const chartOptions = {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{
                    labels: {{
                        color: '#e9ecef',
                        font: {{
                            size: 12
                        }}
                    }}
                }},
                tooltip: {{
                    backgroundColor: 'rgba(39, 43, 48, 0.9)',
                    borderColor: '#7a8288',
                    borderWidth: 1
                }}
            }},
            scales: {{
                x: {{
                    ticks: {{
                        color: '#999999'
                    }},
                    grid: {{
                        color: '#52575c',
                        borderColor: '#7a8288'
                    }}
                }},
                y: {{
                    ticks: {{
                        color: '#999999'
                    }},
                    grid: {{
                        color: '#52575c',
                        borderColor: '#7a8288'
                    }}
                }}
            }}
        }};
        
        // Actual vs Predicted Chart
        const actualVsPredictedCtx = document.getElementById('actualVsPredictedChart').getContext('2d');
        
        // Prepare actual vs predicted data
        const chartDates = [...new Set([
            ...priceData.map(p => p.date),
            ...predictions.filter(p => p.actual).map(p => p.target_date)
        ])].sort();
        
        const actualPrices = chartDates.map(date => {{
            const price = priceData.find(p => p.date === date);
            return price ? price.price : null;
        }});
        
        const predictedPrices = chartDates.map(date => {{
            const pred = predictions.find(p => p.target_date === date && p.actual);
            return pred ? pred.predicted : null;
        }});
        
        new Chart(actualVsPredictedCtx, {{
            type: 'line',
            data: {{
                labels: chartDates.map(d => new Date(d).toLocaleDateString()),
                datasets: [
                    {{
                        label: 'Actual Price',
                        data: actualPrices,
                        borderColor: '#6f42c1',
                        backgroundColor: 'rgba(111, 66, 193, 0.1)',
                        borderWidth: 2,
                        tension: 0.1
                    }},
                    {{
                        label: 'Predicted Price',
                        data: predictedPrices,
                        borderColor: '#e9ecef',
                        backgroundColor: 'rgba(233, 236, 239, 0.1)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        tension: 0.1
                    }}
                ]
            }},
            options: {{
                ...chartOptions,
                scales: {{
                    ...chartOptions.scales,
                    y: {{
                        ...chartOptions.scales.y,
                        ticks: {{
                            ...chartOptions.scales.y.ticks,
                            callback: function(value) {{
                                return '$' + value.toLocaleString();
                            }}
                        }}
                    }}
                }}
            }}
        }});
        
        // Horizon Accuracy Chart
        if (Object.keys(horizonData).length > 0) {{
            const horizonCtx = document.getElementById('horizonAccuracyChart').getContext('2d');
            
            const horizons = Object.keys(horizonData).sort((a, b) => parseInt(a) - parseInt(b));
            const mapeValues = horizons.map(h => horizonData[h].mape);
            const counts = horizons.map(h => horizonData[h].count);
            
            new Chart(horizonCtx, {{
                type: 'bar',
                data: {{
                    labels: horizons,
                    datasets: [{{
                        label: 'Mean Absolute Percentage Error',
                        data: mapeValues,
                        backgroundColor: '#6f42c1',
                        borderColor: '#6f42c1',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    ...chartOptions,
                    scales: {{
                        ...chartOptions.scales,
                        y: {{
                            ...chartOptions.scales.y,
                            ticks: {{
                                ...chartOptions.scales.y.ticks,
                                callback: function(value) {{
                                    return value + '%';
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        }}
        
        // Performance Trend Chart
        if (performanceHistory.length > 0) {{
            const trendCtx = document.getElementById('performanceTrendChart').getContext('2d');
            
            new Chart(trendCtx, {{
                type: 'line',
                data: {{
                    labels: performanceHistory.map(p => new Date(p.date).toLocaleDateString()).reverse(),
                    datasets: [
                        {{
                            label: 'MAPE',
                            data: performanceHistory.map(p => p.mape).reverse(),
                            borderColor: '#e9ecef',
                            backgroundColor: 'rgba(233, 236, 239, 0.1)',
                            borderWidth: 2,
                            tension: 0.3,
                            yAxisID: 'y'
                        }},
                        {{
                            label: 'Directional Accuracy',
                            data: performanceHistory.map(p => p.directional_accuracy * 100).reverse(),
                            borderColor: '#6f42c1',
                            backgroundColor: 'rgba(111, 66, 193, 0.1)',
                            borderWidth: 2,
                            tension: 0.3,
                            yAxisID: 'y1'
                        }}
                    ]
                }},
                options: {{
                    ...chartOptions,
                    scales: {{
                        x: chartOptions.scales.x,
                        y: {{
                            ...chartOptions.scales.y,
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {{
                                display: true,
                                text: 'MAPE (%)',
                                color: '#999999'
                            }}
                        }},
                        y1: {{
                            ...chartOptions.scales.y,
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {{
                                display: true,
                                text: 'Directional Accuracy (%)',
                                color: '#999999'
                            }},
                            grid: {{
                                drawOnChartArea: false
                            }}
                        }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    
    return HTMLResponse(content=html)

def generate_prediction_rows(predictions):
    """Generate HTML rows for predictions table"""
    if not predictions:
        return "<tr><td colspan='5' style='text-align: center; color: var(--medium-gray);'>No predictions for this period</td></tr>"
    
    rows = ""
    for pred in predictions:
        if pred['actual']:
            error_class = "accuracy-good" if pred['error_pct'] < 5 else "accuracy-poor"
            error_text = f"{pred['error_pct']:.1f}%"
        else:
            error_class = ""
            error_text = "—"
        
        actual_display = f"${pred['actual']:,.0f}" if pred['actual'] else "—"
        rows += f"""
        <tr>
            <td>{pred['target_date']}</td>
            <td>${pred['predicted']:,.0f}</td>
            <td>{actual_display}</td>
            <td class="{error_class}">{error_text}</td>
            <td>{pred['horizon']}d</td>
        </tr>
        """
    
    return rows

def generate_signal_rows(signals):
    """Generate HTML rows for signals table"""
    if not signals:
        return "<tr><td colspan='4' style='text-align: center; color: var(--medium-gray);'>No signals for this period</td></tr>"
    
    rows = ""
    for signal in signals:
        signal_class = f"signal-{signal['direction']}"
        strength_display = f"{signal['strength']:+.1f}"
        
        rows += f"""
        <tr>
            <td>{signal['date']}</td>
            <td><span class="signal-badge {signal_class}">{signal['signal'].replace('zen_', '').upper()}</span></td>
            <td>{strength_display}</td>
            <td>{signal['confidence']:.1%}</td>
        </tr>
        """
    
    return rows

if __name__ == "__main__":
    import uvicorn
    print("\n🧘 Starting Zen Consensus Professional Dashboard")
    print("   Visit: http://localhost:8004")
    print("   Press Ctrl+C to stop\n")
    uvicorn.run(app, host="127.0.0.1", port=8004)