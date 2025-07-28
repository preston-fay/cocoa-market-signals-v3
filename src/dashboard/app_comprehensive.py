#!/usr/bin/env python3
"""
Comprehensive Dashboard - Combines interactive timeline with methodology and performance analysis
"""
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import json
import uvicorn
import os

app = FastAPI(title="Cocoa Market Signals - Comprehensive Analysis")
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

# Load dashboard data
with open('data/processed/real_dashboard_data.json', 'r') as f:
    dashboard_data = json.load(f)

# Basic authentication
security = HTTPBasic()

# Get credentials from environment or use defaults
USERNAME = os.environ.get("DASHBOARD_USERNAME", "cocoa")
PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "signals2024")

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify username and password"""
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/", response_class=HTMLResponse)
async def comprehensive_dashboard(request: Request, username: str = Depends(verify_credentials)):
    """Comprehensive dashboard with all features"""
    return templates.TemplateResponse("dashboard_comprehensive.html", {
        "request": request,
        "data": dashboard_data
    })

@app.get("/api/predictions")
async def get_predictions(username: str = Depends(verify_credentials)):
    """Get model predictions"""
    return {
        "predictions": dashboard_data['predictions'],
        "summary": dashboard_data['model_performance']
    }

@app.get("/api/events")
async def get_events(username: str = Depends(verify_credentials)):
    """Get significant market events"""
    return {
        "events": dashboard_data['significant_events'],
        "total": len(dashboard_data['significant_events'])
    }

@app.get("/api/performance")
async def get_performance(username: str = Depends(verify_credentials)):
    """Get detailed model performance"""
    return dashboard_data['model_performance']

@app.get("/api/features")
async def get_features(username: str = Depends(verify_credentials)):
    """Get feature importance"""
    return dashboard_data['feature_importance']

@app.get("/data/processed/{filename}")
async def get_data_file(filename: str, username: str = Depends(verify_credentials)):
    """Serve data files"""
    file_path = f"data/processed/{filename}"
    return FileResponse(file_path)

@app.get("/api/all-prices")
async def get_all_prices(username: str = Depends(verify_credentials)):
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
    import os
    
    print("\n" + "="*60)
    print("🚀 LAUNCHING COMPREHENSIVE DASHBOARD")
    print("="*60)
    print("\nFeatures:")
    print("  ✓ Interactive Timeline with clickable events")
    print("  ✓ Methodology & Data Sources")
    print("  ✓ Model Performance Analysis")
    print("  ✓ Real-time predictions")
    
    # Use PORT from environment or default to 8002
    port = int(os.environ.get("PORT", 8002))
    print(f"\nDashboard URL: http://localhost:{port}")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)