#!/usr/bin/env python3
"""
Zen Consensus Dashboard - STANDARDS COMPLIANT VERSION
NO EMOJIS, NO GRIDLINES, PROPER DARK THEME
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

app = FastAPI(title="Zen Consensus Dashboard - Compliant")

def get_predictions_with_actuals(start_date: date, end_date: date):
    """Get predictions with their actual outcomes"""
    with Session(engine) as session:
        predictions = session.exec(
            select(Prediction)
            .where(and_(
                Prediction.model_name == "zen_consensus",
                Prediction.target_date >= start_date,
                Prediction.target_date <= end_date
            ))
            .order_by(Prediction.target_date)
        ).all()
        
        actuals = session.exec(
            select(PriceData)
            .where(and_(
                PriceData.source == "Yahoo Finance",
                PriceData.date >= start_date,
                PriceData.date <= end_date
            ))
            .order_by(PriceData.date)
        ).all()
        
        actual_prices = {p.date: p.price for p in actuals}
        
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
            'predictions_evaluated': 0,
            'mape': None
        }
    
    mape = np.mean([p['error_pct'] for p in evaluated])
    directional = np.mean([p['direction_correct'] for p in evaluated if p['direction_correct'] is not None])
    avg_confidence = np.mean([p['confidence'] for p in predictions_data])
    
    horizon_accuracy = {}
    for horizon in [1, 7, 30]:
        horizon_preds = [p for p in evaluated if p['horizon'] == horizon]
        if horizon_preds:
            horizon_accuracy[f"{horizon}d"] = {
                'mape': np.mean([p['error_pct'] for p in horizon_preds]),
                'count': len(horizon_preds)
            }
    
    return {
        'overall_accuracy': 100 - mape,
        'mape': mape,
        'directional_accuracy': directional * 100,
        'avg_confidence': avg_confidence * 100,
        'predictions_made': len(predictions_data),
        'predictions_evaluated': len(evaluated),
        'horizon_accuracy': horizon_accuracy
    }

@app.get("/", response_class=HTMLResponse)
async def dashboard(month: str = Query(default=None, description="Month in YYYY-MM format")):
    """Main dashboard with month navigation"""
    
    if month:
        try:
            selected_date = datetime.strptime(month, "%Y-%m").date()
        except:
            selected_date = date.today().replace(day=1)
    else:
        selected_date = date.today().replace(day=1)
    
    start_date = selected_date
    end_date = (selected_date + relativedelta(months=1) - timedelta(days=1))
    
    predictions_data, price_data = get_predictions_with_actuals(start_date, end_date)
    kpis = calculate_kpis(predictions_data)
    
    current_month = selected_date
    prev_month = (current_month - relativedelta(months=1)).strftime("%Y-%m")
    next_month = (current_month + relativedelta(months=1)).strftime("%Y-%m")
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zen Consensus Dashboard - {selected_date.strftime('%B %Y')}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary-purple: #6f42c1;
            --primary-charcoal: #272b30;
            --light-gray: #e9ecef;
            --medium-gray: #999999;
            --dark-gray: #52575c;
            --border-gray: #7a8288;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            background-color: #000000;
            color: var(--light-gray);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            line-height: 1.6;
        }}
        
        .header {{
            background: var(--dark-gray);
            padding: 2rem 0;
            text-align: center;
            border-bottom: 3px solid var(--primary-purple);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            color: var(--light-gray);
            font-weight: 300;
            letter-spacing: 2px;
        }}
        
        .header p {{
            color: var(--medium-gray);
            margin-top: 0.5rem;
        }}
        
        .month-nav {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 2rem;
            padding: 2rem 0;
            background: #000000;
        }}
        
        .btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 500;
            transition: all 0.2s;
            cursor: pointer;
            text-decoration: none;
            border: none;
            outline: none;
            position: relative;
        }}
        
        .btn:focus-visible {{
            outline: 2px solid var(--primary-purple);
            outline-offset: 2px;
        }}
        
        .btn:disabled {{
            pointer-events: none;
            opacity: 0.5;
        }}
        
        .btn-primary {{
            background: var(--primary-purple);
            color: var(--light-gray);
        }}
        
        .btn-primary:hover:not(:disabled) {{
            background: var(--dark-gray);
            transform: translateY(-1px);
        }}
        
        .btn-medium {{
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            height: 2.75rem;
        }}
        
        .current-month {{
            font-size: 1.5rem;
            color: var(--light-gray);
            min-width: 200px;
            text-align: center;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}
        
        .kpi-card {{
            background: var(--primary-charcoal);
            border: 1px solid var(--border-gray);
            padding: 1.5rem;
            text-align: center;
        }}
        
        .kpi-value {{
            font-size: 2.5rem;
            font-weight: 300;
            color: var(--primary-purple);
            margin: 0.5rem 0;
        }}
        
        .kpi-label {{
            color: var(--medium-gray);
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .kpi-subtitle {{
            font-size: 0.875rem;
            color: var(--medium-gray);
            margin-top: 0.5rem;
        }}
        
        .chart-section {{
            background: var(--primary-charcoal);
            border: 1px solid var(--border-gray);
            padding: 2rem;
            margin-bottom: 2rem;
        }}
        
        .chart-title {{
            font-size: 1.5rem;
            color: var(--light-gray);
            margin-bottom: 1.5rem;
            font-weight: 300;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        
        .data-table th {{
            background: var(--dark-gray);
            color: var(--light-gray);
            padding: 1rem;
            text-align: left;
            font-weight: 400;
            text-transform: uppercase;
            font-size: 0.875rem;
            letter-spacing: 1px;
        }}
        
        .data-table td {{
            padding: 1rem;
            border-bottom: 1px solid var(--border-gray);
            color: var(--light-gray);
        }}
        
        .data-table tr:hover {{
            background: var(--dark-gray);
        }}
        
        .accuracy-good {{
            color: var(--primary-purple);
        }}
        
        .accuracy-poor {{
            color: var(--medium-gray);
        }}
        
        .footer {{
            text-align: center;
            padding: 3rem 0;
            color: var(--medium-gray);
            font-size: 0.875rem;
            border-top: 1px solid var(--border-gray);
            margin-top: 4rem;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}
            
            .month-nav {{
                flex-direction: column;
                gap: 1rem;
            }}
            
            .chart-container {{
                height: 300px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ZEN CONSENSUS DASHBOARD</h1>
        <p>AI-Powered Cocoa Market Predictions</p>
    </div>
    
    <div class="month-nav">
        <a href="/?month={prev_month}" class="btn btn-primary btn-medium" aria-label="Navigate to previous month">
            ← Previous Month
        </a>
        <div class="current-month">{selected_date.strftime('%B %Y')}</div>
        <a href="/?month={next_month}" class="btn btn-primary btn-medium" aria-label="Navigate to next month">
            Next Month →
        </a>
    </div>
    
    <div class="container">
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Overall Accuracy</div>
                <div class="kpi-value">{f"{kpis['overall_accuracy']:.1f}%" if kpis.get('overall_accuracy') else "—"}</div>
                <div class="kpi-subtitle">{f"MAPE: {kpis['mape']:.1f}%" if kpis.get('mape') else "No data"}</div>
            </div>
            
            <div class="kpi-card">
                <div class="kpi-label">Directional Accuracy</div>
                <div class="kpi-value">{f"{kpis['directional_accuracy']:.1f}%" if kpis.get('directional_accuracy') else "—"}</div>
                <div class="kpi-subtitle">Correct direction predictions</div>
            </div>
            
            <div class="kpi-card">
                <div class="kpi-label">Predictions Made</div>
                <div class="kpi-value">{kpis.get('predictions_made', 0)}</div>
                <div class="kpi-subtitle">{kpis.get('predictions_evaluated', 0)} evaluated</div>
            </div>
            
            <div class="kpi-card">
                <div class="kpi-label">Average Confidence</div>
                <div class="kpi-value">{f"{kpis['avg_confidence']:.1f}%" if kpis.get('avg_confidence') else "—"}</div>
                <div class="kpi-subtitle">Model confidence score</div>
            </div>
        </div>
        
        <div class="chart-section">
            <h2 class="chart-title">Actual vs Predicted Prices</h2>
            <div class="chart-container">
                <canvas id="actualVsPredictedChart"></canvas>
            </div>
        </div>
        
        <div class="chart-section">
            <h2 class="chart-title">Prediction Details</h2>
            <div style="overflow-x: auto;">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Target Date</th>
                            <th>Predicted</th>
                            <th>Actual</th>
                            <th>Error %</th>
                            <th>Horizon</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {generate_prediction_rows(predictions_data[:15])}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="chart-section">
            <h2 class="chart-title">Accuracy by Prediction Horizon</h2>
            <div class="chart-container" style="height: 300px;">
                <canvas id="horizonChart"></canvas>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Zen Consensus Dashboard • 100% Real Data • Updated Daily</p>
        <p>Following Kearney Design Standards</p>
    </div>
    
    <script>
        // Data preparation
        const predictions = {json.dumps(predictions_data)};
        const priceData = {json.dumps([{'date': p.date.isoformat(), 'price': p.price} for p in price_data])};
        const horizonData = {json.dumps(kpis.get('horizon_accuracy', {}))};
        
        // Chart defaults - NO GRIDLINES, DARK THEME
        Chart.defaults.color = '#999999';
        Chart.defaults.borderColor = '#52575c';
        Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';
        
        // Actual vs Predicted Chart
        const ctx1 = document.getElementById('actualVsPredictedChart').getContext('2d');
        
        // Prepare data with both actual and predicted
        const allDates = [...new Set([
            ...priceData.map(p => p.date),
            ...predictions.map(p => p.target_date)
        ])].sort();
        
        const actualPrices = allDates.map(date => {{
            const price = priceData.find(p => p.date === date);
            return price ? price.price : null;
        }});
        
        const predictedPrices = allDates.map(date => {{
            const pred = predictions.find(p => p.target_date === date);
            return pred ? pred.predicted : null;
        }});
        
        new Chart(ctx1, {{
            type: 'line',
            data: {{
                labels: allDates.map(d => new Date(d).toLocaleDateString('en-US', {{month: 'short', day: 'numeric'}})),
                datasets: [
                    {{
                        label: 'Actual Price',
                        data: actualPrices,
                        borderColor: '#6f42c1',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        pointRadius: 3,
                        pointBackgroundColor: '#6f42c1',
                        tension: 0.2
                    }},
                    {{
                        label: 'Predicted Price',
                        data: predictedPrices,
                        borderColor: '#e9ecef',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        borderDash: [8, 4],
                        pointRadius: 4,
                        pointBackgroundColor: '#e9ecef',
                        tension: 0.2
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false
                }},
                plugins: {{
                    legend: {{
                        labels: {{
                            color: '#e9ecef',
                            padding: 20,
                            font: {{
                                size: 14
                            }}
                        }}
                    }},
                    tooltip: {{
                        backgroundColor: '#000000',
                        titleColor: '#e9ecef',
                        bodyColor: '#999999',
                        borderColor: '#7a8288',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: true,
                        callbacks: {{
                            label: function(context) {{
                                let label = context.dataset.label || '';
                                if (label) {{
                                    label += ': ';
                                }}
                                if (context.parsed.y !== null) {{
                                    label += '$' + context.parsed.y.toLocaleString();
                                }}
                                return label;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        ticks: {{
                            color: '#999999',
                            maxRotation: 45,
                            minRotation: 45
                        }},
                        grid: {{
                            display: false
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
                            display: false
                        }}
                    }}
                }}
            }}
        }});
        
        // Horizon Accuracy Chart
        if (Object.keys(horizonData).length > 0) {{
            const ctx2 = document.getElementById('horizonChart').getContext('2d');
            
            const horizons = Object.keys(horizonData).sort((a, b) => parseInt(a) - parseInt(b));
            const mapeValues = horizons.map(h => horizonData[h].mape);
            
            new Chart(ctx2, {{
                type: 'bar',
                data: {{
                    labels: horizons.map(h => h + ' Horizon'),
                    datasets: [{{
                        label: 'Mean Absolute Percentage Error',
                        data: mapeValues,
                        backgroundColor: '#6f42c1',
                        borderColor: '#6f42c1',
                        borderWidth: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            display: false
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
                                    return 'MAPE: ' + context.parsed.y.toFixed(1) + '%';
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            ticks: {{
                                color: '#999999'
                            }},
                            grid: {{
                                display: false
                            }}
                        }},
                        y: {{
                            ticks: {{
                                color: '#999999',
                                callback: function(value) {{
                                    return value + '%';
                                }}
                            }},
                            grid: {{
                                display: false
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
        return "<tr><td colspan='6' style='text-align: center; color: var(--medium-gray);'>No predictions for this period</td></tr>"
    
    rows = ""
    for pred in predictions:
        if pred['actual']:
            error_class = "accuracy-good" if pred['error_pct'] < 5 else "accuracy-poor"
            error_text = f"{pred['error_pct']:.1f}%"
            actual_text = f"${pred['actual']:,.0f}"
        else:
            error_class = ""
            error_text = "—"
            actual_text = "—"
        
        confidence_text = f"{pred['confidence']*100:.1f}%" if pred['confidence'] else "—"
        
        rows += f"""
        <tr>
            <td>{pred['target_date']}</td>
            <td>${pred['predicted']:,.0f}</td>
            <td>{actual_text}</td>
            <td class="{error_class}">{error_text}</td>
            <td>{pred['horizon']}d</td>
            <td>{confidence_text}</td>
        </tr>
        """
    
    return rows

if __name__ == "__main__":
    import uvicorn
    print("\nStarting Zen Consensus Dashboard (STANDARDS COMPLIANT)")
    print("Visit: http://localhost:8005")
    print("Press Ctrl+C to stop\n")
    uvicorn.run(app, host="127.0.0.1", port=8005)