"""Test different cocoa futures tickers to find working ones"""
import yfinance as yf
from datetime import datetime, timedelta

# Test various cocoa-related tickers
test_tickers = [
    "CC=F",      # Generic cocoa futures
    "CJ=F",      # Cocoa futures
    "NIB.L",     # London cocoa
    "CCU4.NYB",  # ICE Cocoa Sep 2024
    "CC1!",      # Cocoa continuous contract
    "COCOA",     # Generic cocoa
]

end_date = datetime.now()
start_date = end_date - timedelta(days=30)

print("Testing cocoa futures tickers...\n")

for ticker in test_tickers:
    try:
        print(f"Testing {ticker}...")
        data = yf.Ticker(ticker)
        hist = data.history(start=start_date, end=end_date)
        
        if not hist.empty:
            print(f"✓ {ticker} works! Got {len(hist)} days of data")
            print(f"  Latest close: ${hist['Close'].iloc[-1]:.2f}")
            print(f"  Date range: {hist.index[0].date()} to {hist.index[-1].date()}")
        else:
            print(f"✗ {ticker} - No data returned")
    except Exception as e:
        print(f"✗ {ticker} - Error: {str(e)}")
    print()

# Try searching for cocoa-related symbols
print("\nSearching for cocoa-related tickers...")
search_terms = ["cocoa", "cacao", "chocolate"]
for term in search_terms:
    try:
        ticker = yf.Ticker(term)
        info = ticker.info
        if info:
            print(f"Found info for {term}: {info.get('longName', 'N/A')}")
    except Exception:
        pass  # Silently skip if no info found