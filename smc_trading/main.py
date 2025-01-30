from smc_predictor import UniversalSMCPredictor
from utils import get_sp500_tickers

def main():
    smc = UniversalSMCPredictor()
    tickers = get_sp500_tickers()
    
    # Filter out invalid tickers
    valid_tickers = [t for t in tickers[:50] if '.' not in t]  # Remove tickers with dots
    
    try:
        smc.train(valid_tickers, '2010-01-01', '2023-01-01')
        analysis = smc.analyze_market()
        print(analysis.head())
    except ValueError as e:
        print(f"Error in training: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")