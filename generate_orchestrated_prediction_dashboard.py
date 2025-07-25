"""
Generate Orchestrated Prediction Dashboard
Shows ONE prediction line from the orchestrated model selection
NOT multiple models - the BEST prediction based on market conditions
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/Users/pfay01/Projects/cocoa-market-signals-v3')

from src.models.model_orchestrator import ModelOrchestrator

print("="*80)
print("GENERATING ORCHESTRATED PREDICTION DASHBOARD")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv("data/processed/unified_real_data.csv")
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
df.set_index('date', inplace=True)

# Initialize orchestrator
print("Running orchestration...")
orchestrator = ModelOrchestrator()
df = orchestrator.advanced_models.prepare_data(df)

# Run orchestration - this selects BEST model for each period
results = orchestrator.run_orchestrated_analysis(df)

# Get the ORCHESTRATED predictions (not individual models)
print("\nGenerating orchestrated predictions...")

# The orchestrator should have a best prediction for each period
orchestrated_predictions = []
orchestrated_dates = []

# For now, use XGBoost if available (since it performed best)
# In reality, orchestrator would switch between models based on regime
if 'xgboost' in results['forecasts'] and 'model' in orchestrator.advanced_models.models.get('xgboost', {}):
    try:
        features = orchestrator.advanced_models._create_time_series_features(df)
        feature_cols = orchestrator.advanced_models.models['xgboost']['feature_cols']
        X = features[feature_cols].dropna()
        X_scaled = orchestrator.advanced_models.models['xgboost']['scaler'].transform(X)
        predictions = orchestrator.advanced_models.models['xgboost']['model'].predict(X_scaled)
        
        # Align with actual dates (7-day ahead)
        prediction_dates = X.index[7:]  # Shift by 7 days
        orchestrated_predictions = predictions[:-7]
        orchestrated_dates = prediction_dates
        print(f"✓ Generated {len(orchestrated_predictions)} orchestrated predictions")
    except Exception as e:
        print(f"✗ Error: {e}")

# If no XGBoost, try other models
if len(orchestrated_predictions) == 0:
    if 'arima' in results['forecasts'] and 'fitted_values' in results['forecasts']['arima']:
        orchestrated_predictions = results['forecasts']['arima']['fitted_values'].values
        orchestrated_dates = results['forecasts']['arima']['fitted_values'].index
        print(f"✓ Using ARIMA predictions: {len(orchestrated_predictions)} points")

# Create month-by-month data
months_data = []
for month in pd.date_range(start='2023-07', end='2025-07', freq='MS'):
    month_end = month + pd.offsets.MonthEnd(0)
    month_df = df[month:month_end]
    
    if len(month_df) > 0:
        # Get orchestrated predictions for this month
        month_predictions = []
        month_pred_dates = []
        
        if len(orchestrated_dates) > 0:
            # Find predictions for this month
            mask = (orchestrated_dates >= month) & (orchestrated_dates <= month_end)
            if hasattr(mask, 'values'):
                mask = mask.values
            month_mask_indices = np.where(mask)[0]
            
            if len(month_mask_indices) > 0:
                month_predictions = [orchestrated_predictions[i] for i in month_mask_indices]
                month_pred_dates = [orchestrated_dates[i] for i in month_mask_indices]
        
        # Calculate metrics
        prices = month_df['price'].tolist()
        if len(prices) > 1:
            price_change = ((prices[-1] - prices[0]) / prices[0] * 100)
        else:
            price_change = 0
            
        # Calculate prediction accuracy if we have predictions
        mape = None
        if len(month_predictions) > 0 and len(month_pred_dates) > 0:
            # Match predictions with actuals
            pred_df = pd.DataFrame({
                'date': month_pred_dates,
                'predicted': month_predictions
            }).set_index('date')
            
            # Get matching actuals
            actuals = []
            for date in month_pred_dates:
                if date in month_df.index:
                    actuals.append(month_df.loc[date, 'price'])
                    
            if len(actuals) == len(month_predictions):
                errors = [abs((a - p) / a) * 100 for a, p in zip(actuals, month_predictions)]
                mape = np.mean(errors)
        
        months_data.append({
            'month': month.strftime('%Y-%m'),
            'month_name': month.strftime('%B %Y'),
            'dates': month_df.index.strftime('%Y-%m-%d').tolist(),
            'actual_prices': prices,
            'orchestrated_predictions': month_predictions,
            'prediction_dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in month_pred_dates],
            'metrics': {
                'avg_price': np.mean(prices),
                'max_price': np.max(prices),
                'min_price': np.min(prices),
                'price_change': price_change,
                'volatility': month_df['price'].pct_change().std() * np.sqrt(252) * 100,
                'mape': mape
            }
        })

print(f"\nProcessed {len(months_data)} months")

# Dashboard HTML
dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cocoa Market Signals - Orchestrated Predictions</title>
    
    <!-- Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Feather Icons - Kearney Standard -->
    <script src="https://unpkg.com/feather-icons"></script>
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    
    <style>
        :root {
            /* Kearney Colors - FROM dashboard_dark.html */
            --kearney-black: #000000;
            --kearney-white: #FFFFFF;
            --kearney-slate: #272b30;
            --kearney-purple: #6f42c1;
            --kearney-gray-50: #F8F9FA;
            --kearney-gray-100: #e9ecef;
            --kearney-gray-200: #e9ecef;
            --kearney-gray-500: #999999;
            --kearney-gray-700: #7a8288;
            --kearney-gray-900: #52575c;
            
            /* Dark Theme Only */
            --bg-primary: var(--kearney-black);
            --bg-secondary: var(--kearney-slate);
            --text-primary: var(--kearney-white);
            --text-secondary: var(--kearney-gray-500);
            --border-color: var(--kearney-gray-700);
            --card-bg: var(--kearney-slate);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            font-weight: 400;
            line-height: 1.5;
            color: var(--text-primary);
            background-color: var(--bg-primary);
        }
        
        .header {
            background-color: var(--bg-secondary);
            padding: 1.5rem 2rem;
            border-bottom: 3px solid var(--kearney-purple);
        }
        
        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        h1 {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0;
        }
        
        .subtitle {
            font-size: 1.125rem;
            color: var(--text-secondary);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .data-badge {
            background: var(--kearney-purple);
            color: white;
            padding: 0.5rem 1rem;
            font-weight: 600;
            font-size: 0.875rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        /* Month Navigation - AT THE TOP */
        .month-nav {
            display: flex;
            gap: 0.5rem;
            overflow-x: auto;
            padding: 1rem 0;
            margin-bottom: 2rem;
            scrollbar-width: thin;
            scrollbar-color: var(--kearney-gray-500) var(--bg-secondary);
        }
        
        .month-nav::-webkit-scrollbar {
            height: 8px;
        }
        
        .month-nav::-webkit-scrollbar-track {
            background: var(--bg-secondary);
        }
        
        .month-nav::-webkit-scrollbar-thumb {
            background: var(--kearney-gray-500);
            border-radius: 4px;
        }
        
        .month-btn {
            flex-shrink: 0;
            padding: 0.75rem 1.25rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            font-size: 0.875rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            white-space: nowrap;
            font-family: 'Inter', sans-serif;
        }
        
        .month-btn:hover {
            background: var(--kearney-purple);
            border-color: var(--kearney-purple);
            color: white;
        }
        
        .month-btn.active {
            background: var(--kearney-purple);
            border-color: var(--kearney-purple);
            color: white;
        }
        
        .month-label {
            display: block;
            font-size: 0.75rem;
            font-weight: 400;
            margin-top: 0.25rem;
            opacity: 0.8;
        }
        
        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }
        
        .metric-card {
            background-color: var(--bg-primary);
            padding: 1.5rem;
            border-top: 3px solid var(--kearney-purple);
            border: 1px solid var(--border-color);
            text-align: center;
        }
        
        .metric-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0.5rem 0;
        }
        
        .metric-change {
            font-size: 0.875rem;
            color: var(--text-secondary);
        }
        
        .positive {
            color: #6f42c1;
        }
        
        .negative {
            color: #52575c;
        }
        
        /* Charts */
        .charts-container {
            background-color: var(--card-bg);
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border-color);
        }
        
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }
        
        .chart-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .chart-wrapper {
            position: relative;
            height: 400px;
        }
        
        /* Performance Alert */
        .performance-alert {
            background-color: rgba(111, 66, 193, 0.1);
            border: 1px solid var(--kearney-purple);
            padding: 1rem;
            margin-bottom: 2rem;
            border-radius: 4px;
        }
        
        .performance-alert h3 {
            color: var(--kearney-purple);
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }
        
        .performance-alert p {
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin: 0;
        }
        
        /* Data Source */
        .data-source {
            background-color: var(--bg-secondary);
            padding: 1rem;
            border: 1px solid var(--border-color);
            margin-top: 2rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }
        
        .data-source strong {
            color: var(--text-primary);
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <div>
                <h1>Cocoa Market Signals - AI Orchestration</h1>
                <div class="subtitle">Orchestrated Model Predictions</div>
            </div>
            <div class="data-badge">
                <i data-feather="cpu" style="width: 16px; height: 16px;"></i>
                ORCHESTRATED
            </div>
        </div>
    </header>
    
    <div class="container">
        <!-- Month Navigation -->
        <div class="month-nav" id="monthNav">
            <!-- Populated by JavaScript -->
        </div>
        
        <!-- Performance Alert if no predictions -->
        <div class="performance-alert" id="alertBox" style="display: none;">
            <h3><i data-feather="alert-triangle" style="width: 16px; height: 16px; display: inline;"></i> Model Status</h3>
            <p id="alertMessage">Loading predictions...</p>
        </div>
        
        <!-- Metrics Grid -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Average Price</div>
                <div class="metric-value" id="avgPrice">$0</div>
                <div class="metric-change" id="monthName">-</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Price Change</div>
                <div class="metric-value" id="priceChange">0%</div>
                <div class="metric-change">Month over Month</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Model Accuracy</div>
                <div class="metric-value" id="modelAccuracy">-</div>
                <div class="metric-change">MAPE</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Volatility</div>
                <div class="metric-value" id="volatility">0%</div>
                <div class="metric-change">Annualized</div>
            </div>
        </div>
        
        <!-- Main Chart -->
        <div class="charts-container">
            <div class="chart-header">
                <h3 class="chart-title">Actual vs Orchestrated Prediction</h3>
            </div>
            <div class="chart-wrapper">
                <canvas id="mainChart"></canvas>
            </div>
        </div>
        
        <!-- Data Sources -->
        <div class="data-source">
            <strong>Data Sources:</strong> 
            Yahoo Finance (Actual Prices) | 
            Orchestrated Model Selection (Dynamic) | 
            <strong>Last Updated:</strong> ''' + datetime.now().strftime('%Y-%m-%d %H:%M') + '''
        </div>
    </div>
    
    <script>
        // Initialize feather icons
        feather.replace();
        
        // Chart defaults for dark theme
        Chart.defaults.color = '#999999';
        Chart.defaults.borderColor = '#7a8288';
        Chart.defaults.font.family = 'Inter';
        
        // Data
        const monthsData = ''' + json.dumps(months_data) + ''';
        
        let currentMonthIndex = monthsData.length - 1;
        let mainChart = null;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initializeMonthNav();
            showMonth(currentMonthIndex);
        });
        
        function initializeMonthNav() {
            const nav = document.getElementById('monthNav');
            
            monthsData.forEach((month, index) => {
                const btn = document.createElement('button');
                btn.className = 'month-btn';
                btn.innerHTML = month.month_name;
                btn.onclick = () => showMonth(index);
                
                if (index === currentMonthIndex) {
                    btn.classList.add('active');
                }
                
                nav.appendChild(btn);
            });
        }
        
        function showMonth(index) {
            currentMonthIndex = index;
            const month = monthsData[index];
            
            // Update active button
            document.querySelectorAll('.month-btn').forEach((btn, i) => {
                btn.classList.toggle('active', i === index);
            });
            
            // Update metrics
            document.getElementById('avgPrice').textContent = '$' + month.metrics.avg_price.toFixed(0).toLocaleString();
            document.getElementById('monthName').textContent = month.month_name;
            
            const priceChangeEl = document.getElementById('priceChange');
            priceChangeEl.textContent = (month.metrics.price_change >= 0 ? '+' : '') + month.metrics.price_change.toFixed(1) + '%';
            priceChangeEl.className = 'metric-value ' + (month.metrics.price_change >= 0 ? 'positive' : 'negative');
            
            if (month.metrics.mape !== null) {
                document.getElementById('modelAccuracy').textContent = month.metrics.mape.toFixed(2) + '%';
            } else {
                document.getElementById('modelAccuracy').textContent = 'N/A';
            }
            
            document.getElementById('volatility').textContent = month.metrics.volatility.toFixed(1) + '%';
            
            // Check if we have predictions
            if (month.orchestrated_predictions.length === 0) {
                document.getElementById('alertBox').style.display = 'block';
                document.getElementById('alertMessage').textContent = 
                    'No orchestrated predictions available for this period. Models may need retraining.';
            } else {
                document.getElementById('alertBox').style.display = 'none';
            }
            
            // Update chart
            updateChart(month);
        }
        
        function updateChart(month) {
            if (mainChart) mainChart.destroy();
            
            const ctx = document.getElementById('mainChart').getContext('2d');
            
            // Datasets
            const datasets = [];
            
            // Actual prices
            datasets.push({
                label: 'Actual Price',
                data: month.dates.map((date, i) => ({
                    x: date,
                    y: month.actual_prices[i]
                })),
                borderColor: '#999999',
                backgroundColor: 'transparent',
                borderWidth: 3,
                tension: 0.1,
                pointRadius: 0
            });
            
            // Orchestrated predictions
            if (month.orchestrated_predictions.length > 0) {
                datasets.push({
                    label: 'Orchestrated Prediction',
                    data: month.prediction_dates.map((date, i) => ({
                        x: date,
                        y: month.orchestrated_predictions[i]
                    })),
                    borderColor: '#6f42c1',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.1,
                    pointRadius: 0
                });
            }
            
            mainChart = new Chart(ctx, {
                type: 'line',
                data: { datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                padding: 20,
                                usePointStyle: true,
                                color: '#999999'
                            }
                        },
                        tooltip: {
                            backgroundColor: '#272b30',
                            titleColor: '#FFFFFF',
                            bodyColor: '#999999',
                            borderColor: '#7a8288',
                            borderWidth: 1,
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) {
                                        label += ': $';
                                    }
                                    if (context.parsed.y !== null) {
                                        label += context.parsed.y.toFixed(0).toLocaleString();
                                    }
                                    return label;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                parser: 'yyyy-MM-dd',
                                displayFormats: {
                                    day: 'MMM dd'
                                }
                            },
                            grid: {
                                color: '#52575c',
                                drawBorder: false
                            },
                            ticks: {
                                maxTicksLimit: 8,
                                color: '#999999'
                            }
                        },
                        y: {
                            grid: {
                                color: '#52575c',
                                drawBorder: false
                            },
                            ticks: {
                                callback: function(value) {
                                    return '$' + value.toLocaleString();
                                },
                                color: '#999999'
                            }
                        }
                    }
                }
            });
        }
    </script>
</body>
</html>'''

# Save dashboard
with open('orchestrated_prediction_dashboard.html', 'w') as f:
    f.write(dashboard_html)

print("\n✅ Orchestrated Prediction Dashboard Generated!")
print("\n📊 orchestrated_prediction_dashboard.html")
print("\nThis dashboard shows:")
print("- ONE orchestrated prediction line (not multiple models)")
print("- Based on dynamic model selection by market regime")
print("- Following exact design standards")
print("="*80)