#!/usr/bin/env python3
"""
Test Dashboard API endpoints
"""

import requests
import json
from datetime import datetime

def test_dashboard_api():
    """Test the dashboard API endpoints"""
    print("\n" + "="*80)
    print("TESTING ZEN CONSENSUS DASHBOARD API")
    print("="*80 + "\n")
    
    base_url = "http://localhost:8000"
    
    # Test main page
    print("1. Testing main page...")
    try:
        response = requests.get(base_url, timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('content-type')}")
        print("   ✅ Main page accessible")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test consensus API
    print("\n2. Testing /api/consensus endpoint...")
    try:
        response = requests.get(f"{base_url}/api/consensus", timeout=5)
        if response.status_code == 200:
            data = response.json()
            consensus = data['consensus']
            print(f"   Current Price: ${consensus['current_price']:,.2f}")
            print(f"   Forecast: ${consensus['consensus_forecast']:,.2f}")
            print(f"   Signal: {consensus['consensus_signal']}")
            print(f"   Confidence: {consensus['confidence_score']*100:.0f}%")
            print("   ✅ Consensus API working")
        else:
            print(f"   ❌ Status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test signals API
    print("\n3. Testing /api/signals endpoint...")
    try:
        response = requests.get(f"{base_url}/api/signals", timeout=5)
        if response.status_code == 200:
            data = response.json()
            summary = data['summary']
            print(f"   Total Signals: {summary['total_signals']}")
            print(f"   Bullish: {summary['bullish_signals']}")
            print(f"   Bearish: {summary['bearish_signals']}")
            print(f"   Quality: {summary['signal_quality']}")
            print("   ✅ Signals API working")
        else:
            print(f"   ❌ Status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*80)
    print("DASHBOARD API TEST COMPLETE")
    print("\nTo view the dashboard, open: http://localhost:8000")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_dashboard_api()