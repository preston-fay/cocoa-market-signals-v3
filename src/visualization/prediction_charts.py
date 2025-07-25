"""
Enhanced Prediction Charts with Actual vs Predicted
Shows model predictions against actual prices
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Kearney color standards
KEARNEY_COLORS = {
    'purple': '#531E75',
    'gray': '#53565A',
    'blue': '#006FB9',
    'green': '#00A862',
    'red': '#E3001A',
    'orange': '#F47920',
    'light_gray': '#E5E5E5',
    'bg_color': '#F7F7F7'
}

def create_actual_vs_predicted_chart(
    df: pd.DataFrame,
    predictions: Dict[str, pd.Series],
    title: str = "Actual vs Predicted Prices",
    height: int = 500
) -> go.Figure:
    """
    Create chart showing actual prices vs model predictions
    """
    fig = go.Figure()
    
    # Add actual price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['price'],
        mode='lines',
        name='Actual Price',
        line=dict(color=KEARNEY_COLORS['gray'], width=3),
        hovertemplate='Date: %{x}<br>Actual: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add predictions from different models
    colors = [KEARNEY_COLORS['purple'], KEARNEY_COLORS['blue'], KEARNEY_COLORS['green']]
    for i, (model_name, pred_series) in enumerate(predictions.items()):
        if len(pred_series) > 0:
            fig.add_trace(go.Scatter(
                x=pred_series.index,
                y=pred_series,
                mode='lines',
                name=f'{model_name.upper()} Prediction',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                hovertemplate=f'{model_name}: %{{x}}<br>Predicted: $%{{y:,.0f}}<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color=KEARNEY_COLORS['purple'])
        ),
        xaxis=dict(
            title="Date",
            gridcolor=KEARNEY_COLORS['light_gray'],
            showgrid=True
        ),
        yaxis=dict(
            title="Price (USD)",
            gridcolor=KEARNEY_COLORS['light_gray'],
            showgrid=True,
            tickformat='$,.0f'
        ),
        hovermode='x unified',
        height=height,
        plot_bgcolor='white',
        paper_bgcolor=KEARNEY_COLORS['bg_color'],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_prediction_performance_chart(
    performance_data: Dict[str, Dict],
    height: int = 400
) -> go.Figure:
    """
    Create chart showing model prediction performance (MAPE)
    """
    models = list(performance_data.keys())
    mapes = [performance_data[m].get('mape', 0) for m in models]
    
    # Handle empty data
    if not models or not mapes:
        fig = go.Figure()
        fig.add_annotation(
            text="No performance data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color=KEARNEY_COLORS['gray'])
        )
        fig.update_layout(
            height=height,
            plot_bgcolor='white',
            paper_bgcolor=KEARNEY_COLORS['bg_color']
        )
        return fig
    
    # Color based on performance
    colors = []
    for mape in mapes:
        if mape < 1:
            colors.append(KEARNEY_COLORS['green'])
        elif mape < 3:
            colors.append(KEARNEY_COLORS['blue'])
        elif mape < 5:
            colors.append(KEARNEY_COLORS['orange'])
        else:
            colors.append(KEARNEY_COLORS['red'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=models,
        y=mapes,
        marker_color=colors,
        text=[f'{mape:.2f}%' for mape in mapes],
        textposition='outside',
        hovertemplate='%{x}<br>MAPE: %{y:.2f}%<extra></extra>'
    ))
    
    # Add reference line at 3% (good performance threshold)
    fig.add_hline(
        y=3, 
        line_dash="dash", 
        line_color=KEARNEY_COLORS['gray'],
        annotation_text="Good Performance (3%)",
        annotation_position="right"
    )
    
    fig.update_layout(
        title=dict(
            text="Model Prediction Accuracy (Lower is Better)",
            font=dict(size=18, color=KEARNEY_COLORS['purple'])
        ),
        xaxis=dict(
            title="Model",
            tickangle=-45
        ),
        yaxis=dict(
            title="Mean Absolute Percentage Error (%)",
            range=[0, max(mapes) * 1.2]
        ),
        height=height,
        plot_bgcolor='white',
        paper_bgcolor=KEARNEY_COLORS['bg_color'],
        showlegend=False
    )
    
    return fig

def create_forecast_confidence_chart(
    forecasts: Dict[str, Dict],
    current_price: float,
    height: int = 450
) -> go.Figure:
    """
    Create chart showing forecast ranges with confidence intervals
    """
    fig = go.Figure()
    
    # Current price reference
    fig.add_hline(
        y=current_price,
        line_dash="solid",
        line_color=KEARNEY_COLORS['gray'],
        line_width=2,
        annotation_text=f"Current: ${current_price:,.0f}",
        annotation_position="left"
    )
    
    # Add forecasts with confidence intervals
    forecast_days = list(range(1, 31))  # 30-day forecast
    
    for model_name, forecast_data in forecasts.items():
        if 'forecast' in forecast_data:
            forecast_values = forecast_data['forecast']
            
            # Calculate confidence intervals (simplified)
            if 'rmse' in forecast_data:
                rmse = forecast_data['rmse']
                upper_bound = forecast_values + 1.96 * rmse
                lower_bound = forecast_values - 1.96 * rmse
                
                # Add confidence band
                fig.add_trace(go.Scatter(
                    x=forecast_days + forecast_days[::-1],
                    y=list(upper_bound) + list(lower_bound[::-1]),
                    fill='toself',
                    fillcolor=f'rgba({int(KEARNEY_COLORS[list(KEARNEY_COLORS.keys())[0]][1:3], 16)}, '
                              f'{int(KEARNEY_COLORS[list(KEARNEY_COLORS.keys())[0]][3:5], 16)}, '
                              f'{int(KEARNEY_COLORS[list(KEARNEY_COLORS.keys())[0]][5:7], 16)}, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=forecast_days,
                y=forecast_values,
                mode='lines+markers',
                name=model_name.upper(),
                line=dict(width=2),
                hovertemplate=f'{model_name}<br>Day %{{x}}: $%{{y:,.0f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(
            text="30-Day Price Forecasts with Confidence Intervals",
            font=dict(size=18, color=KEARNEY_COLORS['purple'])
        ),
        xaxis=dict(
            title="Days Ahead",
            gridcolor=KEARNEY_COLORS['light_gray'],
            showgrid=True
        ),
        yaxis=dict(
            title="Forecasted Price (USD)",
            gridcolor=KEARNEY_COLORS['light_gray'],
            showgrid=True,
            tickformat='$,.0f'
        ),
        height=height,
        plot_bgcolor='white',
        paper_bgcolor=KEARNEY_COLORS['bg_color'],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    return fig

def create_regime_indicator(
    regime: str,
    volatility: float,
    momentum: float,
    width: int = 400,
    height: int = 200
) -> go.Figure:
    """
    Create visual indicator for current market regime
    """
    # Determine colors and emoji based on regime
    regime_config = {
        'low_volatility': {
            'color': KEARNEY_COLORS['green'],
            'emoji': '🟢',
            'text': 'LOW VOLATILITY'
        },
        'medium_volatility': {
            'color': KEARNEY_COLORS['orange'],
            'emoji': '🟡',
            'text': 'MEDIUM VOLATILITY'
        },
        'high_volatility': {
            'color': KEARNEY_COLORS['red'],
            'emoji': '🔴',
            'text': 'HIGH VOLATILITY'
        }
    }
    
    config = regime_config.get(regime, regime_config['medium_volatility'])
    
    fig = go.Figure()
    
    # Add regime indicator
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=volatility,
        number=dict(
            suffix="%",
            font=dict(size=40, color=config['color'])
        ),
        delta=dict(
            reference=50,
            relative=True,
            font=dict(size=20)
        ),
        title=dict(
            text=f"{config['emoji']} {config['text']}<br><span style='font-size:14px'>Current Volatility</span>",
            font=dict(size=24, color=KEARNEY_COLORS['purple'])
        ),
        domain=dict(x=[0, 0.5], y=[0, 1])
    ))
    
    # Add momentum indicator
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=momentum,
        number=dict(
            suffix="%",
            font=dict(size=40, color=KEARNEY_COLORS['blue'] if momentum > 0 else KEARNEY_COLORS['red'])
        ),
        delta=dict(
            reference=0,
            font=dict(size=20)
        ),
        title=dict(
            text=f"{'📈' if momentum > 0 else '📉'} 5-DAY MOMENTUM<br><span style='font-size:14px'>Price Change</span>",
            font=dict(size=24, color=KEARNEY_COLORS['purple'])
        ),
        domain=dict(x=[0.5, 1], y=[0, 1])
    ))
    
    fig.update_layout(
        height=height,
        width=width,
        paper_bgcolor=KEARNEY_COLORS['bg_color'],
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig

def create_orchestration_dashboard(
    orchestration_results: Dict,
    df: pd.DataFrame
) -> Dict[str, go.Figure]:
    """
    Create complete dashboard for orchestration results
    """
    charts = {}
    
    # 1. Regime indicator
    regime_info = orchestration_results['regime']
    charts['regime_indicator'] = create_regime_indicator(
        regime_info['regime'],
        regime_info['volatility'],
        regime_info['momentum_5d']
    )
    
    # 2. Actual vs Predicted (if we have historical predictions)
    if 'backtest_results' in orchestration_results:
        backtest_df = orchestration_results['backtest_results']
        predictions = {
            'orchestrated': backtest_df['predicted_prices']
        }
        charts['actual_vs_predicted'] = create_actual_vs_predicted_chart(
            backtest_df,
            predictions,
            title="Orchestrated Model: Actual vs Predicted Prices"
        )
    
    # 3. Model performance comparison
    if 'forecasts' in orchestration_results:
        performance_data = {}
        for model_name, result in orchestration_results['forecasts'].items():
            if 'mape' in result:
                performance_data[model_name] = {'mape': result['mape']}
            elif 'cv_scores' in result:
                performance_data[model_name] = {'mape': np.mean(result['cv_scores'])}
        
        if performance_data:
            charts['performance'] = create_prediction_performance_chart(performance_data)
    
    # 4. Forecast with confidence
    if 'forecasts' in orchestration_results:
        current_price = df['price'].iloc[-1]
        charts['forecast_confidence'] = create_forecast_confidence_chart(
            orchestration_results['forecasts'],
            current_price
        )
    
    return charts