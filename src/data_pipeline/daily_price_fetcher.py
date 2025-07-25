"""
Daily Cocoa Price Data Fetcher

Fetches daily cocoa prices from multiple sources for comprehensive analysis.
Extends monitoring period to full year or longer as requested.
"""

import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from loguru import logger
from pathlib import Path

class DailyCocoapriceFetcher:
    """
    Fetches daily cocoa prices from multiple sources
    """
    
    def __init__(self):
        self.data_dir = Path("data/historical/prices")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.add("logs/price_fetcher.log", rotation="1 week")
        
    def fetch_futures_data(self, start_date: str = "2023-01-01", 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch daily cocoa futures data
        
        Tickers:
        - CC=F: Cocoa Futures (Generic)
        - NIB=F: Cocoa Futures (London)
        """
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        logger.info(f"Fetching cocoa futures data from {start_date} to {end_date}")
        
        # Cocoa futures tickers
        # CC=F and CJ=F are the main cocoa futures contracts
        tickers = {
            "cocoa_cc": "CC=F",
            "cocoa_cj": "CJ=F"
        }
        
        all_data = {}
        
        for name, ticker in tickers.items():
            try:
                # Download data using Ticker object for better reliability
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Extract relevant columns
                    price_data = pd.DataFrame({
                        f"{name}_open": data["Open"],
                        f"{name}_high": data["High"],
                        f"{name}_low": data["Low"],
                        f"{name}_close": data["Close"],
                        f"{name}_volume": data["Volume"] if "Volume" in data.columns else 0
                    })
                    all_data[name] = price_data
                    logger.info(f"Successfully fetched {len(data)} days of {name} data")
                else:
                    logger.warning(f"No data available for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {str(e)}")
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data.values(), axis=1)
            combined_df.index.name = "date"
            return combined_df
        else:
            return pd.DataFrame()
    
    def fetch_spot_prices(self) -> Dict:
        """
        Fetch spot prices from various sources
        Note: This would require API keys for commodity data providers
        """
        sources = {
            "quandl": {
                "url": "https://www.quandl.com/api/v3/datasets/CHRIS/ICE_CC1",
                "requires_key": True
            },
            "tradingeconomics": {
                "url": "https://api.tradingeconomics.com/commodity/cocoa",
                "requires_key": True
            }
        }
        
        # Placeholder for when API keys are available
        logger.info("Spot price APIs require authentication")
        return {}
    
    def calculate_daily_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily statistics and technical indicators
        """
        if df.empty:
            return df
            
        # Calculate returns
        for col in df.columns:
            if 'close' in col:
                base_name = col.replace('_close', '')
                df[f'{base_name}_daily_return'] = df[col].pct_change()
                df[f'{base_name}_log_return'] = np.log(df[col] / df[col].shift(1))
                
                # Moving averages
                df[f'{base_name}_ma7'] = df[col].rolling(window=7).mean()
                df[f'{base_name}_ma30'] = df[col].rolling(window=30).mean()
                df[f'{base_name}_ma90'] = df[col].rolling(window=90).mean()
                
                # Volatility
                df[f'{base_name}_volatility_7d'] = df[f'{base_name}_daily_return'].rolling(window=7).std()
                df[f'{base_name}_volatility_30d'] = df[f'{base_name}_daily_return'].rolling(window=30).std()
                
                # RSI
                df[f'{base_name}_rsi'] = self._calculate_rsi(df[col])
                
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def save_daily_data(self, df: pd.DataFrame, filename: str = "daily_cocoa_prices.csv"):
        """
        Save daily price data with metadata
        """
        # Save as CSV
        csv_path = self.data_dir / filename
        df.to_csv(csv_path)
        logger.info(f"Saved daily price data to {csv_path}")
        
        # Also save metadata
        metadata = {
            "source": "Yahoo Finance (Futures)",
            "tickers": ["CC=F", "NIB=F"],
            "start_date": df.index.min().strftime("%Y-%m-%d"),
            "end_date": df.index.max().strftime("%Y-%m-%d"),
            "total_days": len(df),
            "missing_days": df.isnull().sum().to_dict(),
            "last_updated": datetime.now().isoformat(),
            "columns": df.columns.tolist()
        }
        
        metadata_path = self.data_dir / "daily_price_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return csv_path
    
    def fetch_full_year_data(self, years_back: int = 2) -> pd.DataFrame:
        """
        Fetch multiple years of daily data for comprehensive analysis
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years_back)
        
        logger.info(f"Fetching {years_back} years of daily cocoa price data")
        
        # Fetch futures data
        df = self.fetch_futures_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        if not df.empty:
            # Calculate statistics
            df = self.calculate_daily_statistics(df)
            
            # Save the data
            self.save_daily_data(df, f"cocoa_daily_prices_{years_back}yr.csv")
            
            # Create a summary
            self._create_summary_report(df, years_back)
            
        return df
    
    def _create_summary_report(self, df: pd.DataFrame, years: int):
        """
        Create a summary report of the fetched data
        """
        summary = {
            "period": f"{years} years",
            "date_range": {
                "start": df.index.min().strftime("%Y-%m-%d"),
                "end": df.index.max().strftime("%Y-%m-%d")
            },
            "total_trading_days": len(df),
            "price_statistics": {},
            "major_movements": []
        }
        
        # Calculate statistics for each price series
        for col in df.columns:
            if '_close' in col:
                base_name = col.replace('_close', '')
                summary["price_statistics"][base_name] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "current": float(df[col].iloc[-1]),
                    "1yr_return": float((df[col].iloc[-1] / df[col].iloc[-252] - 1) * 100) if len(df) > 252 else None
                }
                
                # Find major movements (>5% daily change)
                returns_col = f'{base_name}_daily_return'
                if returns_col in df.columns:
                    major_moves = df[abs(df[returns_col]) > 0.05]
                    for date, row in major_moves.iterrows():
                        summary["major_movements"].append({
                            "date": date.strftime("%Y-%m-%d"),
                            "ticker": base_name,
                            "change_pct": float(row[returns_col] * 100),
                            "close_price": float(row[col])
                        })
        
        # Sort major movements by absolute change
        summary["major_movements"] = sorted(
            summary["major_movements"], 
            key=lambda x: abs(x["change_pct"]), 
            reverse=True
        )[:20]  # Top 20 movements
        
        # Save summary
        summary_path = self.data_dir / f"daily_price_summary_{years}yr.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Created summary report at {summary_path}")


if __name__ == "__main__":
    import numpy as np  # Import here to avoid issues if not needed elsewhere
    
    # Initialize fetcher
    fetcher = DailyCocoapriceFetcher()
    
    # Fetch 2 years of daily data
    daily_data = fetcher.fetch_full_year_data(years_back=2)
    
    if not daily_data.empty:
        print(f"Successfully fetched {len(daily_data)} days of cocoa price data")
        print(f"\nColumns available: {daily_data.columns.tolist()}")
        print(f"\nLatest prices:")
        print(daily_data.tail())
    else:
        print("Failed to fetch daily price data")