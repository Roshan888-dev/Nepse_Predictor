# utils.py
import requests
import pandas as pd
import yfinance as yf
from typing import List, Dict, Union
from datetime import datetime
import os
from pathlib import Path
from config import PATHS  # Make sure config.py is in the same directory

def get_sp500_tickers() -> List[str]:
    """Fetch current S&P 500 constituents from Wikipedia"""
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        return df['Symbol'].tolist()
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        return []

def download_market_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download historical market data using yfinance"""
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            print(f"No data found for {ticker}")
            return pd.DataFrame()
        return data
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return pd.DataFrame()

def generate_report(results: Dict, filename: Union[str, None] = None) -> None:
    """Generate formatted market analysis report"""
    try:
        # Ensure reports directory exists
        PATHS['reports'].mkdir(parents=True, exist_ok=True)
        
        # Create default filename if not provided
        if not filename:
            filename = f"market_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
        filepath = PATHS['reports'] / filename
        
        # Create formatted report content
        report_content = f"""
        Market Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        {'='*50}
        Total Opportunities: {len(results.get('tickers', []))}
        
        Top 5 Recommendations:
        {pd.DataFrame(results.get('top_picks', [])).to_string(index=False)}
        
        Risk Metrics:
        - Average Volatility: {results.get('avg_volatility', 0):.2%}
        - Maximum Drawdown: {results.get('max_drawdown', 0):.2%}
        - Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}
        """
        
        with open(filepath, 'w') as f:
            f.write(report_content)
            
        print(f"Report generated successfully: {filepath}")
        
    except Exception as e:
        print(f"Error generating report: {e}")