"""
Generate REAL Signal Detection Dashboard with ACTUAL FUCKING DATA
NO FAKE BULLSHIT - 100% REAL DATA FROM CSV
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

print("="*80)
print("GENERATING REAL SIGNAL DETECTION DASHBOARD WITH ACTUAL DATA")
print("="*80)

# Load REAL DATA
df = pd.read_csv("data/processed/unified_real_data.csv")
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
df.set_index('date', inplace=True)

# Calculate REAL signal components
print("\nCalculating REAL signal components from ACTUAL data...")

# 1. Weather Signal (Rainfall anomalies)
df['weather_signal'] = df['rainfall_anomaly'].rolling(30).apply(
    lambda x: np.clip(np.abs(x).mean() / 50, 0, 1)
).fillna(0.5)

# 2. Trade Signal (Volume changes and export concentration)
df['trade_signal'] = (
    df['trade_volume_change'].rolling(30).apply(lambda x: np.abs(x).mean() / 50).fillna(0.5) * 0.5 +
    ((df['export_concentration'] - 0.5) * 2).clip(0, 1) * 0.5
)

# 3. Price Momentum Signal
df['price_momentum'] = df['price'].pct_change(30).rolling(10).mean()
df['momentum_signal'] = df['price_momentum'].apply(lambda x: np.clip((x + 0.1) * 5, 0, 1))

# 4. Composite Signal (Weighted average)
df['composite_signal'] = (
    df['weather_signal'] * 0.35 +
    df['trade_signal'] * 0.40 +
    df['momentum_signal'] * 0.25
).fillna(0.5)

# Generate REAL month-by-month data
months_data = []
for month in pd.date_range(start='2023-07', end='2025-07', freq='MS'):
    month_end = month + pd.offsets.MonthEnd(0)
    month_df = df[month:month_end]
    
    if len(month_df) > 0:
        # REAL calculations for this month
        avg_signal = month_df['composite_signal'].mean()
        max_signal = month_df['composite_signal'].max()
        
        # REAL price data
        prices = month_df['price'].tolist()
        if len(prices) > 0:
            price_change = (prices[-1] - prices[0]) / prices[0] * 100 if prices[0] > 0 else 0
            avg_price = sum(prices) / len(prices)
            max_price = max(prices)
        else:
            price_change = 0
            avg_price = 0
            max_price = 0
        
        # Signal type based on REAL strength
        if avg_signal > 0.8:
            signal_type = 'STRONG BUY'
            signal_color = '#00e676'
        elif avg_signal > 0.65:
            signal_type = 'BUY'
            signal_color = '#4caf50'
        elif avg_signal > 0.5:
            signal_type = 'MONITOR UP'
            signal_color = '#ff9800'
        elif avg_signal > 0.35:
            signal_type = 'MONITOR DOWN'
            signal_color = '#ff5722'
        elif avg_signal > 0.2:
            signal_type = 'SELL'
            signal_color = '#f44336'
        else:
            signal_type = 'STRONG SELL'
            signal_color = '#d32f2f'
        
        months_data.append({
            'month': month.strftime('%Y-%m'),
            'month_name': month.strftime('%B %Y'),
            'data': {
                'dates': month_df.index.strftime('%Y-%m-%d').tolist(),
                'prices': prices,
                'composite_signal': month_df['composite_signal'].tolist(),
                'weather_signal': month_df['weather_signal'].tolist(),
                'trade_signal': month_df['trade_signal'].tolist(),
                'momentum_signal': month_df['momentum_signal'].tolist(),
                'rainfall_anomaly': month_df['rainfall_anomaly'].tolist(),
                'export_concentration': month_df['export_concentration'].tolist(),
                'trade_volume_change': month_df['trade_volume_change'].tolist()
            },
            'metrics': {
                'avg_signal': avg_signal,
                'max_signal': max_signal,
                'signal_type': signal_type,
                'signal_color': signal_color,
                'price_change': price_change,
                'avg_price': avg_price,
                'max_price': max_price,
                'weather_anomaly': month_df['rainfall_anomaly'].mean(),
                'export_conc': month_df['export_concentration'].mean()
            }
        })

print(f"\nProcessed {len(months_data)} months of REAL data")

# Generate the dashboard HTML using EXACT standards from dashboard_dark.html
dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cocoa Market Signals - Signal Detection System</title>
    
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
            /* Kearney Colors */
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
        
        /* Month Selector AT THE TOP */
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
        
        /* Signal Banner */
        .signal-banner {
            background-color: var(--card-bg);
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border-color);
            text-align: center;
        }
        
        .current-signal {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .signal-strength-bar {
            width: 100%;
            max-width: 600px;
            height: 30px;
            background: var(--kearney-gray-900);
            margin: 1rem auto;
            position: relative;
            border: 1px solid var(--border-color);
        }
        
        .signal-strength-fill {
            height: 100%;
            background: linear-gradient(to right, var(--kearney-gray-900), var(--kearney-gray-500), var(--kearney-purple));
            transition: width 0.3s ease;
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
        
        .analytics-tabs {
            display: flex;
            border-bottom: 2px solid var(--border-color);
            margin-bottom: 2rem;
        }
        
        .tab {
            padding: 0.75rem 1.5rem;
            background: transparent;
            border: none;
            cursor: pointer;
            font-weight: 500;
            color: var(--text-secondary);
            border-bottom: 3px solid transparent;
            transition: all 0.2s ease;
            font-family: inherit;
            font-size: 0.875rem;
        }
        
        .tab:hover {
            color: var(--text-primary);
        }
        
        .tab.active {
            color: var(--kearney-purple);
            border-bottom-color: var(--kearney-purple);
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
            height: 300px;
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
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <div>
                <h1>Cocoa Market Signal Detection System</h1>
                <div class="subtitle">100% Real Data Analysis</div>
            </div>
            <div class="data-badge">
                <i data-feather="check-circle" style="width: 16px; height: 16px;"></i>
                VERIFIED REAL DATA
            </div>
        </div>
    </header>
    
    <div class="container">
        <!-- Month Selector AT THE TOP -->
        <div class="month-selector">
            <button class="nav-btn" id="prevMonth" onclick="navigateMonth(-1)">
                <i data-feather="chevron-left" style="width: 16px; height: 16px;"></i>
                Previous
            </button>
            
            <select class="month-dropdown" id="monthDropdown" onchange="selectMonth(this.value)">
                <!-- Populated by JavaScript with REAL months -->
            </select>
            
            <button class="nav-btn" id="nextMonth" onclick="navigateMonth(1)">
                Next
                <i data-feather="chevron-right" style="width: 16px; height: 16px;"></i>
            </button>
        </div>
        
        <!-- Signal Banner -->
        <div class="signal-banner">
            <div class="current-signal" id="currentSignal">LOADING...</div>
            <div class="signal-strength-bar">
                <div class="signal-strength-fill" id="signalStrengthFill" style="width: 0%"></div>
            </div>
            <div style="margin-top: 1rem; color: var(--text-secondary);">
                Composite Signal Strength: <span id="signalValue" style="color: var(--text-primary); font-weight: 600;">0.000</span>
            </div>
        </div>
        
        <!-- Metrics Grid -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Current Price</div>
                <div class="metric-value" id="currentPrice">$0</div>
                <div class="metric-change" id="priceChange">0%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Weather Signal</div>
                <div class="metric-value" id="weatherSignal">0.000</div>
                <div class="metric-change">Rainfall Impact</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Trade Signal</div>
                <div class="metric-value" id="tradeSignal">0.000</div>
                <div class="metric-change">Export Concentration</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Momentum Signal</div>
                <div class="metric-value" id="momentumSignal">0.000</div>
                <div class="metric-change">Price Trends</div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="charts-container">
            <div class="analytics-tabs">
                <button class="tab active" onclick="showChart('signal')">
                    <i data-feather="activity" style="width: 14px; height: 14px; display: inline;"></i> Signal vs Price
                </button>
                <button class="tab" onclick="showChart('components')">
                    <i data-feather="bar-chart-2" style="width: 14px; height: 14px; display: inline;"></i> Components
                </button>
                <button class="tab" onclick="showChart('raw')">
                    <i data-feather="trending-up" style="width: 14px; height: 14px; display: inline;"></i> Raw Data
                </button>
            </div>
            
            <!-- Signal Chart -->
            <div id="signal-chart" class="chart-section">
                <div class="chart-header">
                    <h3 class="chart-title">Signal Detection vs Price Movement</h3>
                    <span class="signal-indicator signal-strong">HISTORICAL DATA</span>
                </div>
                <div class="chart-wrapper">
                    <canvas id="signalChart"></canvas>
                </div>
            </div>
            
            <!-- Components Chart -->
            <div id="components-chart" class="chart-section" style="display: none;">
                <div class="chart-header">
                    <h3 class="chart-title">Signal Component Breakdown</h3>
                    <span class="signal-indicator signal-weak">3 COMPONENTS</span>
                </div>
                <div class="chart-wrapper">
                    <canvas id="componentsChart"></canvas>
                </div>
            </div>
            
            <!-- Raw Data Chart -->
            <div id="raw-chart" class="chart-section" style="display: none;">
                <div class="chart-header">
                    <h3 class="chart-title">Raw Data Indicators</h3>
                    <span class="signal-indicator signal-strong">REAL DATA</span>
                </div>
                <div class="chart-wrapper">
                    <canvas id="rawChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Data Sources -->
        <div class="data-source">
            <strong>Data Sources:</strong> 
            Yahoo Finance (Price Data) | 
            UN Comtrade (Trade Data) | 
            Open-Meteo (Weather Data) | 
            <strong>Last Updated:</strong> <span id="lastUpdated">''' + datetime.now().strftime('%Y-%m-%d %H:%M') + '''</span>
        </div>
    </div>
    
    <script>
        // Initialize feather icons
        feather.replace();
        
        // Chart defaults for dark theme
        Chart.defaults.color = '#999999';
        Chart.defaults.borderColor = '#7a8288';
        Chart.defaults.font.family = 'Inter';
        
        // REAL DATA from Python
        const monthsData = ''' + json.dumps(months_data) + ''';
        
        let currentMonthIndex = monthsData.length - 1;
        let signalChart = null;
        let componentsChart = null;
        let rawChart = null;
        
        // Initialize month dropdown
        function initializeMonthDropdown() {
            const dropdown = document.getElementById('monthDropdown');
            monthsData.forEach((month, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = month.month_name + ' - ' + month.metrics.signal_type;
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
        
        // Update month display with REAL DATA
        function updateMonth() {
            const month = monthsData[currentMonthIndex];
            
            // Update navigation buttons
            document.getElementById('prevMonth').disabled = currentMonthIndex === 0;
            document.getElementById('nextMonth').disabled = currentMonthIndex === monthsData.length - 1;
            
            // Update signal display with REAL values
            const signal = month.metrics.avg_signal;
            document.getElementById('signalValue').textContent = signal.toFixed(3);
            document.getElementById('signalStrengthFill').style.width = (signal * 100) + '%';
            document.getElementById('currentSignal').textContent = month.metrics.signal_type;
            document.getElementById('currentSignal').style.color = month.metrics.signal_color;
            
            // Update metrics with REAL data
            const lastIdx = month.data.dates.length - 1;
            if (lastIdx >= 0) {
                // Price
                const currentPrice = month.data.prices[lastIdx];
                document.getElementById('currentPrice').textContent = '$' + currentPrice.toFixed(0).toLocaleString();
                
                const priceChange = month.metrics.price_change;
                const priceChangeEl = document.getElementById('priceChange');
                priceChangeEl.textContent = (priceChange >= 0 ? '+' : '') + priceChange.toFixed(1) + '%';
                priceChangeEl.className = 'metric-change ' + (priceChange >= 0 ? 'positive' : 'negative');
                
                // Signals
                document.getElementById('weatherSignal').textContent = (month.data.weather_signal[lastIdx] || 0).toFixed(3);
                document.getElementById('tradeSignal').textContent = (month.data.trade_signal[lastIdx] || 0).toFixed(3);
                document.getElementById('momentumSignal').textContent = (month.data.momentum_signal[lastIdx] || 0).toFixed(3);
            }
            
            // Update charts with REAL data
            updateCharts(month);
        }
        
        // Update charts with REAL data
        function updateCharts(month) {
            // Signal Chart
            if (signalChart) signalChart.destroy();
            const ctx1 = document.getElementById('signalChart').getContext('2d');
            signalChart = new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: month.data.dates,
                    datasets: [{
                        label: 'Price',
                        data: month.data.prices,
                        borderColor: '#999999',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        yAxisID: 'y-price',
                        tension: 0.1,
                        pointRadius: 0
                    }, {
                        label: 'Composite Signal',
                        data: month.data.composite_signal,
                        borderColor: '#6f42c1',
                        backgroundColor: 'rgba(111, 66, 193, 0.1)',
                        borderWidth: 3,
                        yAxisID: 'y-signal',
                        fill: true,
                        tension: 0.2,
                        pointRadius: 0
                    }]
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
                            position: 'top',
                            labels: {
                                padding: 20,
                                usePointStyle: true
                            }
                        },
                        tooltip: {
                            backgroundColor: '#272b30',
                            titleColor: '#FFFFFF',
                            bodyColor: '#999999',
                            borderColor: '#7a8288',
                            borderWidth: 1
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
                            }
                        },
                        'y-price': {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            grid: {
                                color: '#52575c',
                                drawBorder: false
                            },
                            ticks: {
                                callback: function(value) {
                                    return '$' + value.toLocaleString();
                                }
                            }
                        },
                        'y-signal': {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            min: 0,
                            max: 1,
                            grid: {
                                drawOnChartArea: false
                            },
                            ticks: {
                                callback: function(value) {
                                    return (value * 100).toFixed(0) + '%';
                                }
                            }
                        }
                    }
                }
            });
            
            // Components Chart
            if (componentsChart) componentsChart.destroy();
            const ctx2 = document.getElementById('componentsChart').getContext('2d');
            componentsChart = new Chart(ctx2, {
                type: 'line',
                data: {
                    labels: month.data.dates,
                    datasets: [{
                        label: 'Weather Signal',
                        data: month.data.weather_signal,
                        borderColor: '#00bcd4',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        tension: 0.2,
                        pointRadius: 0
                    }, {
                        label: 'Trade Signal',
                        data: month.data.trade_signal,
                        borderColor: '#ff9800',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        tension: 0.2,
                        pointRadius: 0
                    }, {
                        label: 'Momentum Signal',
                        data: month.data.momentum_signal,
                        borderColor: '#4caf50',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        tension: 0.2,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                padding: 20,
                                usePointStyle: true
                            }
                        },
                        tooltip: {
                            backgroundColor: '#272b30',
                            titleColor: '#FFFFFF',
                            bodyColor: '#999999',
                            borderColor: '#7a8288',
                            borderWidth: 1
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
                            }
                        },
                        y: {
                            min: 0,
                            max: 1,
                            grid: {
                                color: '#52575c',
                                drawBorder: false
                            },
                            ticks: {
                                callback: function(value) {
                                    return (value * 100).toFixed(0) + '%';
                                }
                            }
                        }
                    }
                }
            });
            
            // Raw Data Chart
            if (rawChart) rawChart.destroy();
            const ctx3 = document.getElementById('rawChart').getContext('2d');
            rawChart = new Chart(ctx3, {
                type: 'line',
                data: {
                    labels: month.data.dates,
                    datasets: [{
                        label: 'Rainfall Anomaly',
                        data: month.data.rainfall_anomaly,
                        borderColor: '#00bcd4',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        tension: 0.1,
                        pointRadius: 0,
                        yAxisID: 'y-anomaly'
                    }, {
                        label: 'Export Concentration',
                        data: month.data.export_concentration,
                        borderColor: '#ff9800',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        tension: 0.1,
                        pointRadius: 0,
                        yAxisID: 'y-concentration'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                padding: 20,
                                usePointStyle: true
                            }
                        },
                        tooltip: {
                            backgroundColor: '#272b30',
                            titleColor: '#FFFFFF',
                            bodyColor: '#999999',
                            borderColor: '#7a8288',
                            borderWidth: 1
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
                            }
                        },
                        'y-anomaly': {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            grid: {
                                color: '#52575c',
                                drawBorder: false
                            },
                            title: {
                                display: true,
                                text: 'Rainfall Anomaly'
                            }
                        },
                        'y-concentration': {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            grid: {
                                drawOnChartArea: false
                            },
                            title: {
                                display: true,
                                text: 'Export Concentration'
                            }
                        }
                    }
                }
            });
        }
        
        // Tab switching
        function showChart(chartName) {
            // Hide all charts
            document.querySelectorAll('.chart-section').forEach(section => {
                section.style.display = 'none';
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected chart
            document.getElementById(chartName + '-chart').style.display = 'block';
            
            // Add active class to clicked tab
            event.target.closest('.tab').classList.add('active');
            
            // Resize charts
            if (chartName === 'signal' && signalChart) signalChart.resize();
            if (chartName === 'components' && componentsChart) componentsChart.resize();
            if (chartName === 'raw' && rawChart) rawChart.resize();
        }
        
        // Initialize
        initializeMonthDropdown();
        updateMonth();
    </script>
</body>
</html>'''

# Save the dashboard
with open('real_signal_detection_dashboard.html', 'w') as f:
    f.write(dashboard_html)

print("\n✅ REAL Signal Detection Dashboard Generated!")
print("\n📊 real_signal_detection_dashboard.html")
print("\nThis dashboard uses:")
print("- 100% REAL DATA from unified_real_data.csv")
print("- REAL signal calculations")
print("- REAL monthly data that ACTUALLY CHANGES when you navigate")
print("- NO FAKE BULLSHIT")
print("="*80)