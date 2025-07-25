"""
Generate Model Performance Dashboard - FOLLOWING EXACT STANDARDS
Shows ACTUAL vs PREDICTED for multiple models
NO SIGNAL DETECTION BULLSHIT
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
from src.models.advanced_time_series_models import AdvancedTimeSeriesModels

print("="*80)
print("GENERATING MODEL PERFORMANCE DASHBOARD - ACTUAL VS PREDICTED")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv("data/processed/unified_real_data.csv")
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
df.set_index('date', inplace=True)

# Initialize models
print("Initializing models...")
orchestrator = ModelOrchestrator()
df = orchestrator.advanced_models.prepare_data(df)

# Run orchestration to get all model predictions
print("Running model orchestration...")
results = orchestrator.run_orchestrated_analysis(df)

# Collect predictions from ALL models
print("\nCollecting predictions from all models...")
all_predictions = {}

# 1. XGBoost predictions
if 'xgboost' in results['forecasts'] and 'model' in orchestrator.advanced_models.models.get('xgboost', {}):
    try:
        features = orchestrator.advanced_models._create_time_series_features(df)
        feature_cols = orchestrator.advanced_models.models['xgboost']['feature_cols']
        X = features[feature_cols].dropna()
        X_scaled = orchestrator.advanced_models.models['xgboost']['scaler'].transform(X)
        xgb_predictions = orchestrator.advanced_models.models['xgboost']['model'].predict(X_scaled)
        
        # Align with actual dates (7-day ahead prediction)
        prediction_dates = X.index[:-7]
        actual_for_predictions = df['price'].loc[X.index[7:]]
        
        all_predictions['XGBoost'] = pd.DataFrame({
            'predicted': xgb_predictions[:-7],
            'actual': actual_for_predictions.values[:len(prediction_dates)]
        }, index=prediction_dates)
        print(f"✓ XGBoost: {len(all_predictions['XGBoost'])} predictions")
    except Exception as e:
        print(f"✗ XGBoost error: {e}")

# 2. ARIMA predictions
if 'arima' in results['forecasts'] and 'fitted_values' in results['forecasts']['arima']:
    arima_fitted = results['forecasts']['arima']['fitted_values']
    all_predictions['ARIMA'] = pd.DataFrame({
        'predicted': arima_fitted,
        'actual': df['price'].loc[arima_fitted.index]
    })
    print(f"✓ ARIMA: {len(all_predictions['ARIMA'])} predictions")

# 3. Holt-Winters predictions
if 'holt_winters' in results['forecasts'] and 'fitted_values' in results['forecasts']['holt_winters']:
    hw_fitted = results['forecasts']['holt_winters']['fitted_values']
    all_predictions['Holt-Winters'] = pd.DataFrame({
        'predicted': hw_fitted,
        'actual': df['price'].loc[hw_fitted.index]
    })
    print(f"✓ Holt-Winters: {len(all_predictions['Holt-Winters'])} predictions")

# 4. LSTM predictions (if available)
if 'lstm_predictor' in results['forecasts'] and 'predictions' in results['forecasts']['lstm_predictor']:
    lstm_preds = results['forecasts']['lstm_predictor']['predictions']
    lstm_dates = results['forecasts']['lstm_predictor']['dates']
    all_predictions['LSTM'] = pd.DataFrame({
        'predicted': lstm_preds,
        'actual': df['price'].loc[lstm_dates]
    }, index=pd.to_datetime(lstm_dates))
    print(f"✓ LSTM: {len(all_predictions['LSTM'])} predictions")

# Month phases with descriptive labels
MONTH_PHASES = {
    '2023-07': {'name': 'July 2023', 'label': 'Baseline Period'},
    '2023-08': {'name': 'August 2023', 'label': 'Stable Market'},
    '2023-09': {'name': 'September 2023', 'label': 'Early Indicators'},
    '2023-10': {'name': 'October 2023', 'label': 'Warning Signs'},
    '2023-11': {'name': 'November 2023', 'label': 'Pattern Formation'},
    '2023-12': {'name': 'December 2023', 'label': 'Signal Strengthening'},
    '2024-01': {'name': 'January 2024', 'label': 'Pre-Surge'},
    '2024-02': {'name': 'February 2024', 'label': 'Surge Begins'},
    '2024-03': {'name': 'March 2024', 'label': 'Rapid Acceleration'},
    '2024-04': {'name': 'April 2024', 'label': 'Peak Volatility'},
    '2024-05': {'name': 'May 2024', 'label': 'Historic Highs'},
    '2024-06': {'name': 'June 2024', 'label': 'Peak Sustained'},
    '2024-07': {'name': 'July 2024', 'label': 'First Correction'},
    '2024-08': {'name': 'August 2024', 'label': 'Volatility Continues'},
    '2024-09': {'name': 'September 2024', 'label': 'Second Peak'},
    '2024-10': {'name': 'October 2024', 'label': 'Stabilization Begins'},
    '2024-11': {'name': 'November 2024', 'label': 'Gradual Decline'},
    '2024-12': {'name': 'December 2024', 'label': 'Year End Stability'},
    '2025-01': {'name': 'January 2025', 'label': 'New Year Reset'},
    '2025-02': {'name': 'February 2025', 'label': 'Steady State'},
    '2025-03': {'name': 'March 2025', 'label': 'Spring Outlook'},
    '2025-04': {'name': 'April 2025', 'label': 'Current Conditions'},
    '2025-05': {'name': 'May 2025', 'label': 'Recent Activity'},
    '2025-06': {'name': 'June 2025', 'label': 'Latest Data'},
    '2025-07': {'name': 'July 2025', 'label': 'Current Month'}
}

# Prepare month-by-month data
months_data = []
for month_key, month_info in MONTH_PHASES.items():
    month_start = pd.to_datetime(month_key + '-01')
    month_end = month_start + pd.offsets.MonthEnd(0)
    
    # Get actual prices for this month
    month_df = df[month_start:month_end]
    
    if len(month_df) > 0:
        # Get predictions for each model for this month
        month_predictions = {}
        
        for model_name, pred_df in all_predictions.items():
            month_pred = pred_df[month_start:month_end]
            if len(month_pred) > 0:
                month_predictions[model_name] = {
                    'dates': month_pred.index.strftime('%Y-%m-%d').tolist(),
                    'predicted': month_pred['predicted'].tolist(),
                    'actual': month_pred['actual'].tolist(),
                    'mape': np.mean(np.abs((month_pred['actual'] - month_pred['predicted']) / month_pred['actual'])) * 100
                }
        
        # Calculate metrics
        prices = month_df['price'].tolist()
        price_change = ((prices[-1] - prices[0]) / prices[0] * 100) if prices[0] > 0 else 0
        
        months_data.append({
            'key': month_key,
            'name': month_info['name'],
            'label': month_info['label'],
            'data': {
                'dates': month_df.index.strftime('%Y-%m-%d').tolist(),
                'actual_prices': prices,
                'predictions': month_predictions
            },
            'metrics': {
                'avg_price': f"${np.mean(prices):,.0f}",
                'max_price': f"${np.max(prices):,.0f}",
                'price_change': f"{price_change:+.1f}%",
                'volatility': f"{month_df['price'].pct_change().std() * np.sqrt(252) * 100:.1f}%",
                'model_count': len(month_predictions)
            }
        })

print(f"\nProcessed {len(months_data)} months of data")
print(f"Models included: {list(all_predictions.keys())}")

# Generate dashboard HTML following EXACT standards
dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cocoa Market Signals - Model Performance</title>
    
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
        
        /* Month Selector - AT THE TOP */
        .month-selector {
            background-color: var(--card-bg);
            padding: 1rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .month-dropdown {
            flex: 1;
            padding: 0.75rem 1rem;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            font-size: 1rem;
            font-family: 'Inter', sans-serif;
            cursor: pointer;
        }
        
        .month-dropdown option {
            background: var(--bg-primary);
            color: var(--text-primary);
        }
        
        .nav-btn {
            padding: 0.75rem 1rem;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: 'Inter', sans-serif;
        }
        
        .nav-btn:hover:not(:disabled) {
            background: var(--kearney-purple);
            border-color: var(--kearney-purple);
            color: white;
        }
        
        .nav-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
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
        
        /* Model Performance Cards */
        .model-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .model-card {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            text-align: center;
        }
        
        .model-name {
            font-weight: 600;
            color: var(--kearney-purple);
            margin-bottom: 0.5rem;
        }
        
        .model-mape {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        
        .best-model {
            border-color: var(--kearney-purple);
            border-width: 2px;
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
        
        /* Signal Indicators */
        .signal-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.25rem 0.75rem;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .signal-strong {
            background-color: var(--kearney-purple);
            color: white;
        }
        
        .signal-weak {
            background-color: var(--kearney-gray-700);
            color: white;
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
        
        /* Legend */
        .custom-legend {
            display: flex;
            gap: 1.5rem;
            justify-content: center;
            margin-top: 1rem;
            flex-wrap: wrap;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .legend-color {
            width: 20px;
            height: 3px;
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <div>
                <h1>Cocoa Market Model Performance</h1>
                <div class="subtitle">Actual vs Predicted Price Analysis</div>
            </div>
            <div class="data-badge">
                <i data-feather="cpu" style="width: 16px; height: 16px;"></i>
                MODEL COMPARISON
            </div>
        </div>
    </header>
    
    <div class="container">
        <!-- Month Selector - AT THE TOP -->
        <div class="month-selector">
            <button class="nav-btn" id="prevMonth" onclick="navigateMonth(-1)">
                <i data-feather="chevron-left" style="width: 16px; height: 16px;"></i>
                Previous
            </button>
            
            <select class="month-dropdown" id="monthDropdown" onchange="selectMonth(this.value)">
                <!-- Populated by JavaScript -->
            </select>
            
            <button class="nav-btn" id="nextMonth" onclick="navigateMonth(1)">
                Next
                <i data-feather="chevron-right" style="width: 16px; height: 16px;"></i>
            </button>
        </div>
        
        <!-- Metrics Grid -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Average Price</div>
                <div class="metric-value" id="avgPrice">$0</div>
                <div class="metric-change" id="monthLabel">-</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Peak Price</div>
                <div class="metric-value" id="maxPrice">$0</div>
                <div class="metric-change">Monthly High</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Price Change</div>
                <div class="metric-value" id="priceChange">0%</div>
                <div class="metric-change">Month over Month</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Volatility</div>
                <div class="metric-value" id="volatility">0%</div>
                <div class="metric-change">Annualized</div>
            </div>
        </div>
        
        <!-- Model Performance Grid -->
        <div class="model-grid" id="modelGrid">
            <!-- Populated by JavaScript -->
        </div>
        
        <!-- Main Chart -->
        <div class="charts-container">
            <div class="chart-header">
                <h3 class="chart-title">Actual vs Predicted Prices</h3>
                <span class="signal-indicator signal-strong" id="modelCount">0 MODELS</span>
            </div>
            <div class="chart-wrapper">
                <canvas id="mainChart"></canvas>
            </div>
            <div class="custom-legend" id="customLegend">
                <!-- Populated by JavaScript -->
            </div>
        </div>
        
        <!-- Data Sources -->
        <div class="data-source">
            <strong>Data Sources:</strong> 
            Yahoo Finance (Actual Prices) | 
            Model Predictions (XGBoost, ARIMA, Holt-Winters, LSTM) | 
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
        
        // REAL DATA
        const monthsData = ''' + json.dumps(months_data) + ''';
        
        // Model colors
        const modelColors = {
            'XGBoost': '#6f42c1',      // Purple
            'ARIMA': '#00bcd4',        // Cyan
            'Holt-Winters': '#ff9800', // Orange
            'LSTM': '#4caf50',         // Green
            'Actual': '#999999'        // Gray
        };
        
        let currentMonthIndex = monthsData.length - 1;
        let mainChart = null;
        
        // Initialize month dropdown
        function initializeMonthDropdown() {
            const dropdown = document.getElementById('monthDropdown');
            monthsData.forEach((month, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `${month.name} - ${month.label}`;
                if (index === currentMonthIndex) {
                    option.selected = true;
                }
                dropdown.appendChild(option);
            });
        }
        
        // Navigate months
        function navigateMonth(direction) {
            const newIndex = currentMonthIndex + direction;
            if (newIndex >= 0 && newIndex < monthsData.length) {
                currentMonthIndex = newIndex;
                document.getElementById('monthDropdown').value = currentMonthIndex;
                updateMonth();
            }
        }
        
        // Select month
        function selectMonth(index) {
            currentMonthIndex = parseInt(index);
            updateMonth();
        }
        
        // Update month display
        function updateMonth() {
            const month = monthsData[currentMonthIndex];
            
            // Update navigation
            document.getElementById('prevMonth').disabled = currentMonthIndex === 0;
            document.getElementById('nextMonth').disabled = currentMonthIndex === monthsData.length - 1;
            
            // Update metrics
            document.getElementById('avgPrice').textContent = month.metrics.avg_price;
            document.getElementById('maxPrice').textContent = month.metrics.max_price;
            document.getElementById('priceChange').textContent = month.metrics.price_change;
            document.getElementById('priceChange').className = 'metric-value ' + 
                (month.metrics.price_change.startsWith('+') ? 'positive' : 'negative');
            document.getElementById('volatility').textContent = month.metrics.volatility;
            document.getElementById('monthLabel').textContent = month.label;
            document.getElementById('modelCount').textContent = month.metrics.model_count + ' MODELS';
            
            // Update model performance cards
            updateModelCards(month);
            
            // Update chart
            updateMainChart(month);
        }
        
        // Update model performance cards
        function updateModelCards(month) {
            const grid = document.getElementById('modelGrid');
            grid.innerHTML = '';
            
            // Calculate MAPE for each model
            const modelMapes = [];
            for (const [modelName, data] of Object.entries(month.data.predictions)) {
                modelMapes.push({
                    name: modelName,
                    mape: data.mape
                });
            }
            
            // Sort by MAPE (best first)
            modelMapes.sort((a, b) => a.mape - b.mape);
            
            // Create cards
            modelMapes.forEach((model, index) => {
                const card = document.createElement('div');
                card.className = 'model-card' + (index === 0 ? ' best-model' : '');
                card.innerHTML = `
                    <div class="model-name">${model.name}</div>
                    <div class="model-mape">${model.mape.toFixed(2)}%</div>
                    <div class="metric-change">MAPE</div>
                `;
                grid.appendChild(card);
            });
        }
        
        // Update main chart
        function updateMainChart(month) {
            if (mainChart) mainChart.destroy();
            
            const ctx = document.getElementById('mainChart').getContext('2d');
            
            // Prepare datasets
            const datasets = [];
            
            // Add actual prices
            datasets.push({
                label: 'Actual',
                data: month.data.actual_prices,
                borderColor: modelColors['Actual'],
                backgroundColor: 'transparent',
                borderWidth: 3,
                tension: 0.1,
                pointRadius: 0
            });
            
            // Add model predictions
            for (const [modelName, data] of Object.entries(month.data.predictions)) {
                datasets.push({
                    label: modelName,
                    data: data.predicted,
                    borderColor: modelColors[modelName] || '#ffffff',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.1,
                    pointRadius: 0
                });
            }
            
            // Create chart
            mainChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: month.data.dates,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            display: false // Using custom legend
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
                                maxTicksLimit: 8
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
                                }
                            }
                        }
                    }
                }
            });
            
            // Update custom legend
            updateCustomLegend(datasets);
        }
        
        // Update custom legend
        function updateCustomLegend(datasets) {
            const legend = document.getElementById('customLegend');
            legend.innerHTML = '';
            
            datasets.forEach(dataset => {
                const item = document.createElement('div');
                item.className = 'legend-item';
                
                const color = document.createElement('div');
                color.className = 'legend-color';
                color.style.backgroundColor = dataset.borderColor;
                if (dataset.borderDash) {
                    color.style.backgroundImage = `repeating-linear-gradient(90deg, ${dataset.borderColor} 0px, ${dataset.borderColor} 5px, transparent 5px, transparent 10px)`;
                }
                
                const label = document.createElement('span');
                label.textContent = dataset.label;
                label.style.color = dataset.label === 'Actual' ? '#ffffff' : '#999999';
                
                item.appendChild(color);
                item.appendChild(label);
                legend.appendChild(item);
            });
        }
        
        // Initialize
        initializeMonthDropdown();
        updateMonth();
    </script>
</body>
</html>'''

# Save dashboard
with open('model_performance_dashboard.html', 'w') as f:
    f.write(dashboard_html)

print("\n✅ Model Performance Dashboard Generated!")
print("\n📊 model_performance_dashboard.html")
print("\nThis dashboard shows:")
print("- ACTUAL vs PREDICTED prices for multiple models")
print("- Real model predictions from XGBoost, ARIMA, Holt-Winters, LSTM")
print("- Month-by-month navigation with REAL changing data")
print("- Model performance metrics (MAPE)")
print("- Following EXACT design standards")
print("="*80)