"""
Data Status Component
Shows real-time status of data sources and API connections
"""

from dash import html, dcc
import plotly.graph_objects as go
import os
from datetime import datetime
from typing import Dict, Any

def check_api_status() -> Dict[str, Any]:
    """Check status of all configured APIs"""
    status = {
        'price_data': {
            'commodities_api': {
                'configured': bool(os.getenv('COMMODITIES_API_KEY')),
                'name': 'Commodities-API',
                'status': 'configured' if os.getenv('COMMODITIES_API_KEY') else 'missing'
            },
            'alpha_vantage': {
                'configured': bool(os.getenv('ALPHA_VANTAGE_API_KEY')),
                'name': 'Alpha Vantage',
                'status': 'configured' if os.getenv('ALPHA_VANTAGE_API_KEY') else 'missing'
            },
            'twelve_data': {
                'configured': bool(os.getenv('TWELVE_DATA_API_KEY')),
                'name': 'Twelve Data',
                'status': 'configured' if os.getenv('TWELVE_DATA_API_KEY') else 'missing'
            }
        },
        'weather_data': {
            'noaa': {
                'configured': bool(os.getenv('NOAA_API_KEY')),
                'name': 'NOAA Climate',
                'status': 'configured' if os.getenv('NOAA_API_KEY') else 'missing'
            },
            'openweather': {
                'configured': bool(os.getenv('OPENWEATHER_API_KEY')),
                'name': 'OpenWeatherMap',
                'status': 'configured' if os.getenv('OPENWEATHER_API_KEY') else 'missing'
            }
        },
        'trade_data': {
            'comtrade': {
                'configured': bool(os.getenv('COMTRADE_API_KEY')),
                'name': 'UN Comtrade',
                'status': 'configured' if os.getenv('COMTRADE_API_KEY') else 'missing'
            }
        }
    }
    
    # Count configured APIs
    total_apis = 0
    configured_apis = 0
    
    for category in status.values():
        for api in category.values():
            total_apis += 1
            if api['configured']:
                configured_apis += 1
    
    status['summary'] = {
        'total': total_apis,
        'configured': configured_apis,
        'percentage': (configured_apis / total_apis * 100) if total_apis > 0 else 0
    }
    
    return status

def create_data_status_card():
    """Create a card showing data source status"""
    status = check_api_status()
    
    # Create status indicator
    if status['summary']['percentage'] == 100:
        overall_status = "✅ All Data Sources Configured"
        status_color = "#6f42c1"  # Kearney purple
    elif status['summary']['percentage'] > 0:
        overall_status = f"⚠️ {status['summary']['configured']}/{status['summary']['total']} Data Sources Configured"
        status_color = "#999999"  # Kearney medium gray
    else:
        overall_status = "❌ No Data Sources Configured"
        status_color = "#52575c"  # Kearney dark gray
    
    # Build status rows
    status_rows = []
    
    # Price Data Status
    status_rows.append(
        html.Div([
            html.H5("📈 Price Data APIs", style={'marginBottom': '10px'}),
            html.Div([
                create_api_status_row(api['name'], api['status'])
                for api in status['price_data'].values()
            ])
        ], style={'marginBottom': '20px'})
    )
    
    # Weather Data Status
    status_rows.append(
        html.Div([
            html.H5("🌦️ Weather Data APIs", style={'marginBottom': '10px'}),
            html.Div([
                create_api_status_row(api['name'], api['status'])
                for api in status['weather_data'].values()
            ])
        ], style={'marginBottom': '20px'})
    )
    
    # Trade Data Status
    status_rows.append(
        html.Div([
            html.H5("📊 Trade Data APIs", style={'marginBottom': '10px'}),
            html.Div([
                create_api_status_row(api['name'], api['status'])
                for api in status['trade_data'].values()
            ])
        ], style={'marginBottom': '20px'})
    )
    
    return html.Div([
        html.Div([
            html.H3("Data Source Status", style={'marginBottom': '20px'}),
            html.Div([
                html.H4(overall_status, style={'color': status_color}),
                html.P(f"Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            ], style={'marginBottom': '30px'}),
            html.Div(status_rows),
            html.Div([
                html.Hr(),
                html.P([
                    "📖 See ",
                    html.A("DATA_DISCLAIMER.md", href="/DATA_DISCLAIMER.md", target="_blank"),
                    " for setup instructions"
                ]),
                html.P([
                    "⚠️ ",
                    html.Strong("Important: "),
                    "This system requires real data sources. Performance metrics are calculated from actual data."
                ], style={'color': '#52575c'})  # Kearney dark gray
            ])
        ], style={
            'padding': '20px',
            'backgroundColor': '#e9ecef',  # Kearney light gray
            'borderRadius': '10px',
            'marginBottom': '30px',
            'border': '1px solid #7a8288'  # Kearney border gray
        })
    ])

def create_api_status_row(name: str, status: str) -> html.Div:
    """Create a single API status row"""
    if status == 'configured':
        icon = "✅"
        color = "#6f42c1"  # Kearney purple
        text = "Configured"
    else:
        icon = "❌"
        color = "#52575c"  # Kearney dark gray
        text = "Not Configured"
    
    return html.Div([
        html.Span(f"{icon} {name}: ", style={'fontWeight': 'bold'}),
        html.Span(text, style={'color': color})
    ], style={'marginBottom': '5px', 'paddingLeft': '20px'})

def create_data_disclaimer_banner():
    """Create a banner warning about data requirements"""
    return html.Div([
        html.Div([
            html.Strong("⚠️ IMPORTANT DATA NOTICE"),
            html.P([
                "This system requires REAL DATA from external APIs to function properly. ",
                "All performance metrics are calculated from actual data inputs. ",
                "Please configure your API keys in the .env file. ",
                html.A("View Setup Instructions", href="/DATA_DISCLAIMER.md", 
                      style={'color': 'white', 'textDecoration': 'underline'})
            ], style={'marginBottom': '0'})
        ], style={
            'backgroundColor': '#52575c',  # Kearney dark gray
            'color': 'white',
            'padding': '15px',
            'borderRadius': '5px',
            'marginBottom': '20px',
            'textAlign': 'center'
        })
    ]) if check_api_status()['summary']['percentage'] < 100 else None