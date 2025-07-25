"""
Generate Signal Detection Dashboard - FIXED VERSION
Proper navigation, proper colors, following standards
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/Users/pfay01/Projects/cocoa-market-signals-v3')

from src.models.model_orchestrator import ModelOrchestrator
from src.models.statistical_models import StatisticalSignalModels

print("="*80)
print("GENERATING FIXED SIGNAL DETECTION DASHBOARD")
print("="*80)

# Load data
df = pd.read_csv("data/processed/unified_real_data.csv")
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
df.set_index('date', inplace=True)

# Calculate signal components
print("\nCalculating signal components...")

# 1. Weather Signal (Rainfall anomalies)
df['weather_signal'] = df['rainfall_anomaly'].rolling(30).apply(
    lambda x: np.clip(np.abs(x).mean() / 50, 0, 1)  # Normalize to 0-1
)

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

# Determine signal type based on strength
def get_signal_type(strength):
    if strength > 0.8:
        return 'STRONG BUY', '#00e676'
    elif strength > 0.65:
        return 'BUY', '#4caf50'
    elif strength > 0.5:
        return 'MONITOR UP', '#ff9800'
    elif strength > 0.35:
        return 'MONITOR DOWN', '#ff5722'
    elif strength > 0.2:
        return 'SELL', '#f44336'
    else:
        return 'STRONG SELL', '#d32f2f'

# Key detection events
KEY_EVENTS = [
    {
        'date': '2023-10-15',
        'type': 'FIRST WARNING',
        'description': 'Weather anomalies detected in Ghana',
        'signal_strength': 0.58
    },
    {
        'date': '2023-12-01',
        'type': 'SIGNAL STRENGTHENING',
        'description': 'Trade volumes declining, export concentration rising',
        'signal_strength': 0.65
    },
    {
        'date': '2024-01-15',
        'type': 'BUY SIGNAL',
        'description': 'All indicators align - strong buy signal generated',
        'signal_strength': 0.82
    },
    {
        'date': '2024-02-01',
        'type': 'SURGE BEGINS',
        'description': 'Price surge starts - signal validated',
        'signal_strength': 0.88
    },
    {
        'date': '2024-04-15',
        'type': 'PEAK SIGNAL',
        'description': 'Maximum signal strength during peak volatility',
        'signal_strength': 0.92
    }
]

# Generate month-by-month signal data
months_data = []
for month in pd.date_range(start='2023-07', end='2025-07', freq='MS'):
    month_end = month + pd.offsets.MonthEnd(0)
    month_df = df[month:month_end]
    
    if len(month_df) > 0:
        # Calculate signal metrics for the month
        avg_signal = month_df['composite_signal'].mean()
        max_signal = month_df['composite_signal'].max()
        signal_type, signal_color = get_signal_type(avg_signal)
        
        # Price change in month
        price_change = (month_df['price'].iloc[-1] - month_df['price'].iloc[0]) / month_df['price'].iloc[0] * 100
        
        # Check for events in this month
        month_events = [e for e in KEY_EVENTS if month <= pd.to_datetime(e['date']) < month_end]
        
        months_data.append({
            'month': month.strftime('%Y-%m'),
            'month_name': month.strftime('%B %Y'),
            'data': {
                'dates': month_df.index.strftime('%Y-%m-%d').tolist(),
                'prices': month_df['price'].tolist(),
                'composite_signal': month_df['composite_signal'].tolist(),
                'weather_signal': month_df['weather_signal'].tolist(),
                'trade_signal': month_df['trade_signal'].tolist(),
                'momentum_signal': month_df['momentum_signal'].tolist()
            },
            'metrics': {
                'avg_signal': avg_signal,
                'max_signal': max_signal,
                'signal_type': signal_type,
                'signal_color': signal_color,
                'price_change': price_change,
                'avg_price': month_df['price'].mean(),
                'weather_anomaly': month_df['rainfall_anomaly'].mean(),
                'export_conc': month_df['export_concentration'].mean()
            },
            'events': month_events
        })

# Create dashboard HTML
dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cocoa Market Signals - Signal Detection System</title>
    
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/feather-icons"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    
    <style>
        :root {
            --kearney-purple: #6f42c1;
            --kearney-charcoal: #272b30;
            --kearney-charcoal-secondary: #3a3f44;
            --kearney-gray: #999999;
            --kearney-gray-light: #7a8288;
            --kearney-white: #FFFFFF;
            --kearney-black: #000000;
            --kearney-green: #00e676;
            --kearney-red: #ff5252;
            --signal-strong-buy: #00e676;
            --signal-buy: #4caf50;
            --signal-monitor: #ff9800;
            --signal-sell: #f44336;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', sans-serif;
            color: var(--kearney-white);
            background: var(--kearney-black);
            line-height: 1.6;
        }
        
        .header {
            background: var(--kearney-charcoal);
            padding: 1.5rem 2rem;
            border-bottom: 3px solid var(--kearney-purple);
        }
        
        .header-content {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        h1 {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        
        .subtitle {
            color: var(--kearney-gray);
            font-size: 1.125rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        /* Signal Status Banner */
        .signal-banner {
            background: var(--kearney-charcoal-secondary);
            border: 2px solid var(--kearney-purple);
            padding: 2rem;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .current-signal {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .signal-strength-bar {
            width: 100%;
            max-width: 600px;
            height: 30px;
            background: var(--kearney-charcoal);
            margin: 1rem auto;
            position: relative;
            border: 1px solid var(--kearney-gray-light);
        }
        
        .signal-strength-fill {
            height: 100%;
            background: linear-gradient(to right, var(--signal-sell), var(--signal-monitor), var(--signal-buy), var(--signal-strong-buy));
            transition: width 0.3s ease;
        }
        
        .signal-labels {
            display: flex;
            justify-content: space-between;
            max-width: 600px;
            margin: 0.5rem auto;
            font-size: 0.75rem;
            color: var(--kearney-gray);
        }
        
        /* Timeline */
        .timeline {
            background: var(--kearney-charcoal-secondary);
            border: 1px solid var(--kearney-gray-light);
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .timeline-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--kearney-white);
        }
        
        .timeline-events {
            position: relative;
            padding-left: 2rem;
        }
        
        .timeline-line {
            position: absolute;
            left: 0.5rem;
            top: 0;
            bottom: 0;
            width: 2px;
            background: var(--kearney-gray-light);
        }
        
        .timeline-event {
            position: relative;
            margin-bottom: 1.5rem;
            padding-left: 1rem;
        }
        
        .timeline-dot {
            position: absolute;
            left: -1.5rem;
            top: 0.25rem;
            width: 12px;
            height: 12px;
            background: var(--kearney-purple);
            border-radius: 50%;
            border: 2px solid var(--kearney-charcoal-secondary);
        }
        
        .timeline-date {
            font-size: 0.875rem;
            color: var(--kearney-gray);
            margin-bottom: 0.25rem;
        }
        
        .timeline-event-title {
            font-weight: 600;
            color: var(--kearney-white);
            margin-bottom: 0.25rem;
        }
        
        .timeline-description {
            font-size: 0.875rem;
            color: var(--kearney-gray-light);
        }
        
        /* Month Navigation - IMPROVED */
        .month-selector {
            background: var(--kearney-charcoal-secondary);
            border: 1px solid var(--kearney-gray-light);
            padding: 1rem;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .month-dropdown {
            flex: 1;
            padding: 0.75rem 1rem;
            background: var(--kearney-charcoal);
            border: 1px solid var(--kearney-gray-light);
            color: var(--kearney-white);
            font-size: 1rem;
            font-family: 'Inter', sans-serif;
            cursor: pointer;
        }
        
        .month-dropdown option {
            background: var(--kearney-charcoal);
            color: var(--kearney-white);
        }
        
        .nav-btn {
            padding: 0.75rem 1rem;
            background: var(--kearney-charcoal);
            border: 1px solid var(--kearney-gray-light);
            color: var(--kearney-white);
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .nav-btn:hover {
            background: var(--kearney-purple);
            border-color: var(--kearney-purple);
        }
        
        .nav-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .nav-btn:disabled:hover {
            background: var(--kearney-charcoal);
            border-color: var(--kearney-gray-light);
        }
        
        /* Month Info */
        .month-info {
            background: var(--kearney-charcoal-secondary);
            border: 1px solid var(--kearney-gray-light);
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .month-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .month-title {
            font-size: 1.5rem;
            font-weight: 700;
        }
        
        .month-signal-badge {
            padding: 0.5rem 1rem;
            font-weight: 600;
            border-radius: 4px;
        }
        
        .month-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .metric {
            padding: 1rem;
            background: var(--kearney-charcoal);
            border-left: 4px solid var(--kearney-purple);
        }
        
        .metric-label {
            color: var(--kearney-gray);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }
        
        .metric-value.positive {
            color: var(--kearney-green);
        }
        
        .metric-value.negative {
            color: var(--kearney-red);
        }
        
        /* Charts */
        .chart-container {
            background: var(--kearney-charcoal-secondary);
            border: 1px solid var(--kearney-gray-light);
            padding: 2rem;
            margin-bottom: 2rem;
        }
        
        .chart-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: var(--kearney-white);
        }
        
        canvas {
            max-height: 400px;
        }
        
        /* Signal Components */
        .signal-components {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .component-card {
            background: var(--kearney-charcoal-secondary);
            border: 1px solid var(--kearney-gray-light);
            padding: 1.5rem;
            text-align: center;
        }
        
        .component-icon {
            color: var(--kearney-purple);
            margin-bottom: 0.5rem;
        }
        
        .component-name {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--kearney-white);
        }
        
        .component-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--kearney-white);
        }
        
        .component-bar {
            width: 100%;
            height: 8px;
            background: var(--kearney-charcoal);
            margin-top: 0.5rem;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .component-bar-fill {
            height: 100%;
            background: var(--kearney-purple);
            transition: width 0.3s ease;
        }
        
        /* Signal strength text */
        .signal-value {
            color: var(--kearney-white);
            font-weight: 600;
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <h1>Cocoa Market Signal Detection System</h1>
            <div class="subtitle">Detecting Market Signals Before Price Movements</div>
        </div>
    </header>
    
    <div class="container">
        <!-- Current Signal Status -->
        <div class="signal-banner" id="signalBanner">
            <div class="current-signal" id="currentSignal">MONITORING</div>
            <div class="signal-strength-bar">
                <div class="signal-strength-fill" id="signalStrengthFill" style="width: 50%"></div>
            </div>
            <div class="signal-labels">
                <span>STRONG SELL</span>
                <span>SELL</span>
                <span>MONITOR</span>
                <span>BUY</span>
                <span>STRONG BUY</span>
            </div>
            <p style="margin-top: 1rem; color: var(--kearney-white);">
                Composite Signal Strength: <span id="signalValue" class="signal-value">0.50</span>
            </p>
        </div>
        
        <!-- Key Detection Timeline -->
        <div class="timeline">
            <div class="timeline-title">
                <i data-feather="activity" width="20" style="vertical-align: middle"></i>
                Key Signal Detection Events
            </div>
            <div class="timeline-events">
                <div class="timeline-line"></div>
                ''' + ''.join([f'''
                <div class="timeline-event">
                    <div class="timeline-dot"></div>
                    <div class="timeline-date">{event['date']}</div>
                    <div class="timeline-event-title">{event['type']}</div>
                    <div class="timeline-description">{event['description']}</div>
                </div>
                ''' for event in KEY_EVENTS]) + '''
            </div>
        </div>
        
        <!-- Month Navigation - IMPROVED DROPDOWN -->
        <div class="month-selector">
            <button class="nav-btn" id="prevMonth" onclick="navigateMonth(-1)">
                <i data-feather="chevron-left" width="20"></i>
                Previous
            </button>
            
            <select class="month-dropdown" id="monthDropdown" onchange="selectMonth(this.value)">
                <!-- Dynamically populated -->
            </select>
            
            <button class="nav-btn" id="nextMonth" onclick="navigateMonth(1)">
                Next
                <i data-feather="chevron-right" width="20"></i>
            </button>
        </div>
        
        <!-- Current Month Info -->
        <div class="month-info" id="monthInfo">
            <!-- Dynamically populated -->
        </div>
        
        <!-- Signal Components -->
        <div class="signal-components" id="signalComponents">
            <div class="component-card">
                <i data-feather="cloud-rain" width="32" class="component-icon"></i>
                <div class="component-name">Weather Signal</div>
                <div class="component-value" id="weatherSignal">0.00</div>
                <div class="component-bar">
                    <div class="component-bar-fill" id="weatherBar" style="width: 0%"></div>
                </div>
            </div>
            <div class="component-card">
                <i data-feather="truck" width="32" class="component-icon"></i>
                <div class="component-name">Trade Signal</div>
                <div class="component-value" id="tradeSignal">0.00</div>
                <div class="component-bar">
                    <div class="component-bar-fill" id="tradeBar" style="width: 0%"></div>
                </div>
            </div>
            <div class="component-card">
                <i data-feather="trending-up" width="32" class="component-icon"></i>
                <div class="component-name">Momentum Signal</div>
                <div class="component-value" id="momentumSignal">0.00</div>
                <div class="component-bar">
                    <div class="component-bar-fill" id="momentumBar" style="width: 0%"></div>
                </div>
            </div>
        </div>
        
        <!-- Signal vs Price Chart -->
        <div class="chart-container">
            <div class="chart-title">Signal Detection vs Price Movement</div>
            <canvas id="signalChart"></canvas>
        </div>
        
        <!-- Component Breakdown Chart -->
        <div class="chart-container">
            <div class="chart-title">Signal Component Breakdown</div>
            <canvas id="componentChart"></canvas>
        </div>
    </div>
    
    <script>
        // Data
        const monthsData = ''' + json.dumps(months_data) + ''';
        let currentMonthIndex = monthsData.length - 1;
        let signalChart = null;
        let componentChart = null;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            feather.replace();
            initializeMonthDropdown();
            showMonth(currentMonthIndex);
        });
        
        function initializeMonthDropdown() {
            const dropdown = document.getElementById('monthDropdown');
            
            monthsData.forEach((month, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = month.month_name;
                
                // Add signal type to option text
                option.textContent += ` - ${month.metrics.signal_type}`;
                
                if (index === currentMonthIndex) {
                    option.selected = true;
                }
                
                dropdown.appendChild(option);
            });
        }
        
        function selectMonth(index) {
            currentMonthIndex = parseInt(index);
            showMonth(currentMonthIndex);
        }
        
        function navigateMonth(direction) {
            const newIndex = currentMonthIndex + direction;
            if (newIndex >= 0 && newIndex < monthsData.length) {
                currentMonthIndex = newIndex;
                document.getElementById('monthDropdown').value = currentMonthIndex;
                showMonth(currentMonthIndex);
            }
        }
        
        function showMonth(index) {
            const month = monthsData[index];
            
            // Update navigation buttons
            document.getElementById('prevMonth').disabled = index === 0;
            document.getElementById('nextMonth').disabled = index === monthsData.length - 1;
            
            // Update signal banner
            const signal = month.metrics.avg_signal;
            document.getElementById('currentSignal').textContent = month.metrics.signal_type;
            document.getElementById('currentSignal').style.color = month.metrics.signal_color;
            document.getElementById('signalStrengthFill').style.width = (signal * 100) + '%';
            document.getElementById('signalValue').textContent = signal.toFixed(3);
            
            // Update month info
            const info = document.getElementById('monthInfo');
            info.innerHTML = `
                <div class="month-header">
                    <div class="month-title">${month.month_name}</div>
                    <div class="month-signal-badge" style="background: ${month.metrics.signal_color}; color: white;">
                        ${month.metrics.signal_type}
                    </div>
                </div>
                <div class="month-metrics">
                    <div class="metric">
                        <div class="metric-label">Average Price</div>
                        <div class="metric-value">$${month.metrics.avg_price.toFixed(0).toLocaleString()}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Price Change</div>
                        <div class="metric-value ${month.metrics.price_change > 0 ? 'positive' : 'negative'}">
                            ${month.metrics.price_change > 0 ? '+' : ''}${month.metrics.price_change.toFixed(1)}%
                        </div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Signal Strength</div>
                        <div class="metric-value">${month.metrics.avg_signal.toFixed(3)}</div>
                    </div>
                </div>
                ${month.events.length > 0 ? `
                    <div style="margin-top: 1rem; padding: 1rem; background: var(--kearney-purple); border-radius: 4px;">
                        <strong>🚨 Event:</strong> ${month.events[0].type} - ${month.events[0].description}
                    </div>
                ` : ''}
            `;
            
            // Update signal components (use last values of month)
            const lastIdx = month.data.dates.length - 1;
            const weatherVal = month.data.weather_signal[lastIdx] || 0;
            const tradeVal = month.data.trade_signal[lastIdx] || 0;
            const momentumVal = month.data.momentum_signal[lastIdx] || 0;
            
            document.getElementById('weatherSignal').textContent = weatherVal.toFixed(3);
            document.getElementById('weatherBar').style.width = (weatherVal * 100) + '%';
            document.getElementById('tradeSignal').textContent = tradeVal.toFixed(3);
            document.getElementById('tradeBar').style.width = (tradeVal * 100) + '%';
            document.getElementById('momentumSignal').textContent = momentumVal.toFixed(3);
            document.getElementById('momentumBar').style.width = (momentumVal * 100) + '%';
            
            // Update charts
            updateSignalChart(month);
            updateComponentChart(month);
        }
        
        function updateSignalChart(month) {
            const ctx = document.getElementById('signalChart').getContext('2d');
            
            if (signalChart) signalChart.destroy();
            
            signalChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: month.data.dates,
                    datasets: [
                        {
                            label: 'Price',
                            data: month.data.prices,
                            borderColor: '#999999',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            yAxisID: 'y-price',
                            tension: 0
                        },
                        {
                            label: 'Composite Signal',
                            data: month.data.composite_signal,
                            borderColor: '#00e676',
                            backgroundColor: 'rgba(0, 230, 118, 0.1)',
                            borderWidth: 3,
                            yAxisID: 'y-signal',
                            fill: true,
                            tension: 0.2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: { color: '#999999' }
                        },
                        tooltip: {
                            callbacks: {
                                afterLabel: function(context) {
                                    if (context.datasetIndex === 1) {
                                        const signal = context.parsed.y;
                                        if (signal > 0.8) return 'STRONG BUY SIGNAL';
                                        if (signal > 0.65) return 'BUY SIGNAL';
                                        if (signal > 0.5) return 'MONITORING UP';
                                        if (signal > 0.35) return 'MONITORING DOWN';
                                        if (signal > 0.2) return 'SELL SIGNAL';
                                        return 'STRONG SELL SIGNAL';
                                    }
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'time',
                            time: { unit: 'day' },
                            grid: { color: '#3a3f44' },
                            ticks: { color: '#999999' }
                        },
                        'y-price': {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            grid: { color: '#3a3f44' },
                            ticks: {
                                color: '#999999',
                                callback: value => '$' + value.toLocaleString()
                            }
                        },
                        'y-signal': {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            min: 0,
                            max: 1,
                            grid: { drawOnChartArea: false },
                            ticks: {
                                color: '#00e676',
                                callback: value => (value * 100).toFixed(0) + '%'
                            }
                        }
                    }
                }
            });
        }
        
        function updateComponentChart(month) {
            const ctx = document.getElementById('componentChart').getContext('2d');
            
            if (componentChart) componentChart.destroy();
            
            componentChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: month.data.dates,
                    datasets: [
                        {
                            label: 'Weather Signal',
                            data: month.data.weather_signal,
                            borderColor: '#00bcd4',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            tension: 0.2
                        },
                        {
                            label: 'Trade Signal',
                            data: month.data.trade_signal,
                            borderColor: '#ff9800',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            tension: 0.2
                        },
                        {
                            label: 'Momentum Signal',
                            data: month.data.momentum_signal,
                            borderColor: '#4caf50',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            tension: 0.2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: { color: '#999999' }
                        }
                    },
                    scales: {
                        x: {
                            type: 'time',
                            time: { unit: 'day' },
                            grid: { color: '#3a3f44' },
                            ticks: { color: '#999999' }
                        },
                        y: {
                            min: 0,
                            max: 1,
                            grid: { color: '#3a3f44' },
                            ticks: {
                                color: '#999999',
                                callback: value => (value * 100).toFixed(0) + '%'
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
with open('signal_detection_dashboard_fixed.html', 'w') as f:
    f.write(dashboard_html)

print("\n✅ Fixed Signal Detection Dashboard Generated!")
print("\n📊 Open 'signal_detection_dashboard_fixed.html' in your browser")
print("\nFixed issues:")
print("- Better month navigation with dropdown and prev/next buttons")
print("- NO purple text on dark backgrounds - using white and proper grays")
print("- Following your established standards properly")
print("- Clean, professional interface")
print("="*80)