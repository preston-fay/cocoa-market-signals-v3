#!/usr/bin/env python3
"""
Real Dashboard - Showing ACTUAL model results
100% Kearney compliant, 100% real data
"""
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import pandas as pd
from datetime import datetime

app = FastAPI(title="Cocoa Market Signals - Real Results")
app.mount("/static", StaticFiles(directory="."), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="templates")

# Load real dashboard data
with open('data/processed/real_dashboard_data.json', 'r') as f:
    dashboard_data = json.load(f)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard showing real results"""
    return templates.TemplateResponse("dashboard_working.html", {
        "request": request,
        "data": dashboard_data
    })

@app.get("/api/predictions")
async def get_predictions():
    """Get model predictions"""
    return {
        "predictions": dashboard_data['predictions'],  # ALL predictions
        "summary": dashboard_data['model_performance']
    }

@app.get("/api/events")
async def get_events():
    """Get significant market events"""
    return {
        "events": dashboard_data['significant_events'],  # ALL events
        "total": len(dashboard_data['significant_events'])
    }

@app.get("/api/performance")
async def get_performance():
    """Get detailed model performance"""
    return dashboard_data['model_performance']

@app.get("/api/features")
async def get_features():
    """Get feature importance"""
    return dashboard_data['feature_importance']

@app.get("/data/processed/{filename}")
async def get_data_file(filename: str):
    """Serve data files"""
    file_path = f"data/processed/{filename}"
    return FileResponse(file_path)

@app.get("/timeline", response_class=HTMLResponse)
async def timeline_dashboard(request: Request):
    """Interactive timeline dashboard"""
    return templates.TemplateResponse("timeline_fixed_final.html", {
        "request": request
    })

@app.get("/api/all-prices")
async def get_all_prices():
    """Get all price data for complete timeline"""
    import sqlite3
    import pandas as pd
    
    conn = sqlite3.connect('data/cocoa_market_signals_real.db')
    query = """
    SELECT 
        p1.date,
        p1.close as price,
        p2.close as prev_price,
        (p1.close - p2.close) / p2.close as return
    FROM price_data p1
    LEFT JOIN price_data p2 ON date(p2.date) = date(p1.date, '-1 day')
    ORDER BY p1.date
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Convert to list of dicts
    prices = []
    for _, row in df.iterrows():
        prices.append({
            "date": row['date'] + 'T00:00:00',
            "price": float(row['price']),
            "return": float(row['return']) if pd.notna(row['return']) else 0
        })
    
    return {"prices": prices}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 LAUNCHING REAL RESULTS DASHBOARD")
    print("="*60)
    print("\nModel Performance:")
    for model, metrics in dashboard_data['model_performance']['all_models'].items():
        print(f"  {model}: {metrics['accuracy']:.1%} accuracy")
    print(f"\nDashboard URL: http://localhost:8001")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)