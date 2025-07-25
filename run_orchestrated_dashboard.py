"""
Run Orchestrated Dashboard with Actual vs Predicted
Shows model predictions, orchestration, and performance
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.visualization.prediction_charts import (
    create_actual_vs_predicted_chart,
    create_prediction_performance_chart,
    create_forecast_confidence_chart,
    create_regime_indicator,
    KEARNEY_COLORS
)
from src.models.model_orchestrator import ModelOrchestrator
# from src.data_processing.unified_data_processor import UnifiedDataProcessor  # Not needed

# Initialize Dash app
app = dash.Dash(__name__)

# Load data
print("Loading unified real data...")
df = pd.read_csv("data/processed/unified_real_data.csv")
df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
df.set_index('date', inplace=True)

# Initialize orchestrator
print("Initializing orchestrator...")
orchestrator = ModelOrchestrator()
df = orchestrator.advanced_models.prepare_data(df)

# Run initial orchestration
print("Running orchestrated analysis...")
orchestration_results = orchestrator.run_orchestrated_analysis(df)

# Create layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🎯 Cocoa Market Signals - AI Orchestration Dashboard", 
                style={'color': KEARNEY_COLORS['purple'], 'textAlign': 'center'}),
        html.P("Dynamic Model Selection with Actual vs Predicted Performance", 
               style={'textAlign': 'center', 'fontSize': '18px', 'color': KEARNEY_COLORS['gray']})
    ], style={'backgroundColor': KEARNEY_COLORS['bg_color'], 'padding': '20px'}),
    
    # Top Row: Regime and Signal
    html.Div([
        html.Div([
            dcc.Graph(
                id='regime-indicator',
                figure=create_regime_indicator(
                    orchestration_results['regime']['regime'],
                    orchestration_results['regime']['volatility'],
                    orchestration_results['regime']['momentum_5d']
                )
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Div([
                html.H3("📊 Trading Signal", style={'color': KEARNEY_COLORS['purple']}),
                html.Div([
                    html.H2(
                        f"{orchestration_results['signals']['direction'].upper()}", 
                        style={
                            'color': KEARNEY_COLORS['green'] if orchestration_results['signals']['direction'] == 'bullish' 
                                    else KEARNEY_COLORS['red'] if orchestration_results['signals']['direction'] == 'bearish'
                                    else KEARNEY_COLORS['gray'],
                            'margin': '10px'
                        }
                    ),
                    html.P(f"Signal Strength: {orchestration_results['signals']['signal_strength']:.0%}"),
                    html.P(f"Confidence: {orchestration_results['signals']['confidence']:.0%}"),
                    html.P(f"Risk Level: {orchestration_results['signals']['risk_level'].upper()}")
                ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px'})
            ])
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ], style={'margin': '20px'}),
    
    # Model Performance
    html.Div([
        html.H3("🏆 Model Performance Comparison", style={'color': KEARNEY_COLORS['purple'], 'marginLeft': '20px'}),
        dcc.Graph(
            id='model-performance',
            figure=create_prediction_performance_chart({
                model: {'mape': result.get('mape', np.mean(result.get('cv_scores', [5])))}
                for model, result in orchestration_results['forecasts'].items()
            })
        )
    ], style={'margin': '20px'}),
    
    # Actual vs Predicted Chart
    html.Div([
        html.H3("📈 Real-Time Predictions", style={'color': KEARNEY_COLORS['purple'], 'marginLeft': '20px'}),
        dcc.Graph(id='actual-vs-predicted'),
        dcc.Interval(id='interval-component', interval=5000, n_intervals=0)  # Update every 5 seconds
    ], style={'margin': '20px'}),
    
    # Forecast Confidence
    html.Div([
        html.H3("🔮 30-Day Forecast with Confidence Intervals", style={'color': KEARNEY_COLORS['purple'], 'marginLeft': '20px'}),
        dcc.Graph(
            id='forecast-confidence',
            figure=create_forecast_confidence_chart(
                orchestration_results['forecasts'],
                df['price'].iloc[-1]
            )
        )
    ], style={'margin': '20px'}),
    
    # Recommendations
    html.Div([
        html.H3("💡 AI Recommendations", style={'color': KEARNEY_COLORS['purple']}),
        html.Div([
            html.Ul([
                html.Li(rec, style={'fontSize': '16px', 'marginBottom': '10px'})
                for rec in orchestration_results['recommendations']
            ])
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px'})
    ], style={'margin': '20px'}),
    
    # Selected Models
    html.Div([
        html.H3("🤖 Active Models", style={'color': KEARNEY_COLORS['purple']}),
        html.Div([
            html.H4("Forecast Models:", style={'color': KEARNEY_COLORS['gray']}),
            html.P(', '.join(orchestration_results['models_used']['forecast']).upper()),
            html.H4("Anomaly Detection:", style={'color': KEARNEY_COLORS['gray']}),
            html.P(', '.join(orchestration_results['models_used']['anomaly']).upper())
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px'})
    ], style={'margin': '20px', 'width': '48%', 'display': 'inline-block'})
    
], style={'backgroundColor': KEARNEY_COLORS['bg_color']})

# Callback for actual vs predicted
@app.callback(
    Output('actual-vs-predicted', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_actual_vs_predicted(n):
    # Get latest predictions
    predictions = {}
    
    # XGBoost prediction (if available)
    if 'xgboost' in orchestration_results['forecasts'] and 'model' in orchestrator.advanced_models.models.get('xgboost', {}):
        try:
            # Get last 90 days for visualization
            recent_df = df.iloc[-90:]
            features = orchestrator.advanced_models._create_time_series_features(recent_df)
            feature_cols = orchestrator.advanced_models.models['xgboost']['feature_cols']
            
            # Make predictions for historical data
            X = features[feature_cols].dropna()
            X_scaled = orchestrator.advanced_models.models['xgboost']['scaler'].transform(X)
            xgb_predictions = orchestrator.advanced_models.models['xgboost']['model'].predict(X_scaled)
            
            # Align with dates (accounting for lag)
            pred_dates = X.index
            predictions['XGBoost'] = pd.Series(xgb_predictions, index=pred_dates)
        except:
            pass
    
    # Add other model predictions if available
    for model_name in ['arima', 'sarima', 'holt_winters']:
        if model_name in orchestration_results['forecasts']:
            try:
                model_result = orchestration_results['forecasts'][model_name]
                if 'fitted_values' in model_result:
                    predictions[model_name] = model_result['fitted_values'][-90:]
            except:
                pass
    
    return create_actual_vs_predicted_chart(
        df.iloc[-90:],
        predictions,
        title="Model Predictions vs Actual Prices (Last 90 Days)"
    )

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 ORCHESTRATED DASHBOARD RUNNING")
    print("="*80)
    print(f"\nCurrent Market Regime: {orchestration_results['regime']['regime'].replace('_', ' ').upper()}")
    print(f"Active Models: {', '.join(orchestration_results['models_used']['forecast'])}")
    print(f"Signal: {orchestration_results['signals']['direction'].upper()}")
    print(f"\nDashboard available at: http://127.0.0.1:8050")
    print("="*80)
    
    app.run_server(debug=True)