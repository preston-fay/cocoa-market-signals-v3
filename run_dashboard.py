#!/usr/bin/env python3
"""
Run the Market Signal Detection Dashboard
HTMX + FastAPI implementation following Kearney standards
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MARKET SIGNAL DETECTION POC - INTERACTIVE DASHBOARD")
    print("="*60)
    print("\nThis dashboard demonstrates how AI and advanced analytics")
    print("predicted the 2024 cocoa price surge with 3 months lead time.")
    print("\nTechnology Stack:")
    print("- Frontend: HTMX (dynamic HTML over the wire)")
    print("- Backend: FastAPI")
    print("- Charts: Plotly.js")
    print("- Design: Kearney standards")
    print("\nStarting server...")
    print("Open http://localhost:8053 in your browser")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        import uvicorn
        from src.dashboard.app_fastapi import app
        
        uvicorn.run(app, host="0.0.0.0", port=8053, log_level="info")
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")
    except ImportError as e:
        print(f"\nError: Missing dependency - {e}")
        print("\nPlease install requirements:")
        print("  pip install fastapi uvicorn")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease check the logs for details.")