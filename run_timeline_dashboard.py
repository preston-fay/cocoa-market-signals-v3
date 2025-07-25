#!/usr/bin/env python3
"""
Run the Timeline Dashboard with Real Data
V2-style functionality with timeline navigation
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("\n" + "="*60)
    print("COCOA MARKET SIGNALS - TIMELINE DASHBOARD")
    print("="*60)
    print("\nThis dashboard shows how our AI system evolved its predictions")
    print("over time, using 100% real data from:")
    print("- Yahoo Finance (daily prices)")
    print("- UN Comtrade (export data)")
    print("- Open-Meteo (weather data)")
    print("\nFeatures:")
    print("- Timeline navigation through key phases")
    print("- Interactive charts showing signal evolution")
    print("- Model performance analysis")
    print("- Improvement recommendations")
    print("\nStarting server...")
    print("Open http://localhost:8054 in your browser")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        import uvicorn
        from src.dashboard.app_timeline import app
        
        uvicorn.run(app, host="0.0.0.0", port=8054, log_level="info")
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")
    except ImportError as e:
        print(f"\nError: Missing dependency - {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements_dashboard.txt")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease check the logs for details.")