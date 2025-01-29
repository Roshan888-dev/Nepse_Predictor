import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf  # Added yfinance import

# ======================
# Data Requirements
# ======================
"""
Required Data Structure (from yfinance):
- Date (Index)
- Open
- High
- Low
- Close
- Volume
- VWAP (Calculated as (H+L+C)/3)
"""

# ======================
# SMC Core Components
# ======================
class SMC_Analyzer:
    def __init__(self):
        self.market_structure = []
        self.order_blocks = []
        self.liquidity_zones = []
        self.imbalances = []
        self.supply_demand_zones = []
        self.wyckoff_phases = []
    
    # Market Structure Analysis (unchanged)
    def analyze_market_structure(self, df):
        structure = []
        highs = df['High']
        lows = df['Low']
        
        for i in range(2, len(df)-2):
            # Break of Structure (BOS)
            if highs[i] > highs[i-2] and highs[i] > highs[i-1]:
                structure.append(('BOS', 'bullish', df.index[i]))
            elif lows[i] < lows[i-2] and lows[i] < lows[i-1]:
                structure.append(('BOS', 'bearish', df.index[i]))
            
            # Change of Character (CHOCH)
            if (highs[i] < highs[i-1] and highs[i-1] > highs[i-2] and 
                lows[i] > lows[i-1] and lows[i-1] < lows[i-2]):
                structure.append(('CHOCH', 'bearish', df.index[i]))
            elif (lows[i] > lows[i-1] and lows[i-1] < lows[i-2] and 
                  highs[i] < highs[i-1] and highs[i-1] > highs[i-2]):
                structure.append(('CHOCH', 'bullish', df.index[i]))
        
        self.market_structure = structure
        return structure
    
    # Order Block Detection (unchanged)
    def identify_order_blocks(self, df):
        blocks = []
        for i in range(2, len(df)):
            # Bullish Order Block
            if (df['Close'].iloc[i-2] < df['Open'].iloc[i-2] and
                df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and
                df['Close'].iloc[i] > df['Open'].iloc[i] and
                df['Volume'].iloc[i] > df['Volume'].iloc[i-1]*1.5):
                blocks.append(('Bullish OB', df.index[i], df['Low'].iloc[i]))
            
            # Bearish Order Block
            if (df['Close'].iloc[i-2] > df['Open'].iloc[i-2] and
                df['Close'].iloc[i-1] > df['Open'].iloc[i-1] and
                df['Close'].iloc[i] < df['Open'].iloc[i] and
                df['Volume'].iloc[i] > df['Volume'].iloc[i-1]*1.5):
                blocks.append(('Bearish OB', df.index[i], df['High'].iloc[i]))
        
        self.order_blocks = blocks
        return blocks
    
    # Liquidity Zone Identification (unchanged)
    def find_liquidity_zones(self, df):
        swing_highs = df['High'][(df['High'] > df['High'].shift(1)) & 
                                (df['High'] > df['High'].shift(-1))]
        swing_lows = df['Low'][(df['Low'] < df['Low'].shift(1)) & 
                              (df['Low'] < df['Low'].shift(-1))]
        
        self.liquidity_zones = {
            'swing_highs': swing_highs.tolist(),
            'swing_lows': swing_lows.tolist(),
            'equal_highs': df[df['High'] == df['High'].shift(1)]['High'].tolist(),
            'equal_lows': df[df['Low'] == df['Low'].shift(1)]['Low'].tolist()
        }
        return self.liquidity_zones
    
    # Fair Value Gap Detection (unchanged)
    def detect_imbalances(self, df):
        imbalances = []
        for i in range(1, len(df)-1):
            current_low = df['Low'].iloc[i]
            current_high = df['High'].iloc[i]
            next_high = df['High'].iloc[i+1]
            next_low = df['Low'].iloc[i+1]
            
            # Bullish Imbalance
            if current_high < next_low:
                imbalances.append(('Bullish FVG', df.index[i], 
                                 current_high, next_low))
            
            # Bearish Imbalance
            if current_low > next_high:
                imbalances.append(('Bearish FVG', df.index[i], 
                                 next_high, current_low))
        
        self.imbalances = imbalances
        return imbalances
    
    # Wyckoff Phase Analysis (unchanged)
    def analyze_wyckoff(self, df):
        phases = []
        vwap = df['VWAP'] if 'VWAP' in df.columns else df['Close']
        
        for i in range(4, len(df)-4):
            # Accumulation Phase
            if (df['Volume'].iloc[i] > df['Volume'].iloc[i-1] and
                df['Close'].iloc[i] > vwap.iloc[i] and
                df['Low'].iloc[i] > df['Low'].iloc[i-2]):
                phases.append(('Accumulation', df.index[i]))
            
            # Distribution Phase
            if (df['Volume'].iloc[i] > df['Volume'].iloc[i-1] and
                df['Close'].iloc[i] < vwap.iloc[i] and
                df['High'].iloc[i] < df['High'].iloc[i-2]):
                phases.append(('Distribution', df.index[i]))
        
        self.wyckoff_phases = phases
        return phases

# ======================
# Apple Trading System
# ======================
class Apple_SMC_TradingSystem:  # Renamed class
    def __init__(self):
        self.data = None
        self.analyzer = SMC_Analyzer()
    
    def fetch_data(self, ticker, start_date, end_date):  # Modified data loading
        self.data = yf.download(ticker, start=start_date, end=end_date)
        # Calculate VWAP
        self.data['VWAP'] = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        return self.data
    
    def full_analysis(self):
        self.analyzer.analyze_market_structure(self.data)
        self.analyzer.identify_order_blocks(self.data)
        self.analyzer.find_liquidity_zones(self.data)
        self.analyzer.detect_imbalances(self.data)
        self.analyzer.analyze_wyckoff(self.data)
        return self.data
    
    def generate_signals(self):  # Unchanged
        signals = []
        for idx, row in self.data.iterrows():
            signal = None
            price = row['Close']
            
            # Check Order Blocks
            for block in self.analyzer.order_blocks:
                if block[1] == idx:
                    if block[0] == 'Bullish OB' and price <= block[2]:
                        signal = 'Buy'
                    elif block[0] == 'Bearish OB' and price >= block[2]:
                        signal = 'Sell'
            
            # Check Liquidity Zones
            for zone in self.analyzer.liquidity_zones['swing_highs']:
                if abs(price - zone) < price*0.005:
                    signal = 'Sell'
            for zone in self.analyzer.liquidity_zones['swing_lows']:
                if abs(price - zone) < price*0.005:
                    signal = 'Buy'
            
            # Check Imbalances
            for imb in self.analyzer.imbalances:
                if imb[0] == 'Bullish FVG' and imb[2] <= price <= imb[3]:
                    signal = 'Buy'
                elif imb[0] == 'Bearish FVG' and imb[2] <= price <= imb[3]:
                    signal = 'Sell'
            
            signals.append(signal)
        
        self.data['Signal'] = signals
        return self.data
    def calculate_accuracy(self, horizons=[1, 3, 5]):
        if 'Signal' not in self.data.columns:
            raise ValueError("Generate signals first using generate_signals()")
            
        accuracy = {
            'Buy': {h: {'success': 0, 'total': 0} for h in horizons},
            'Sell': {h: {'success': 0, 'total': 0} for h in horizons}
        }

        signals = self.data[self.data['Signal'].notnull()]
        
        for idx, row in signals.iterrows():
            signal_type = row['Signal']
            pos = self.data.index.get_loc(idx)
            
            for horizon in horizons:
                if pos + horizon >= len(self.data):
                    continue
                
                future_prices = self.data['Close'].iloc[pos+1:pos+horizon+1]
                current_price = row['Close']
                
                if signal_type == 'Buy':
                    success = any(future_prices > current_price)
                elif signal_type == 'Sell':
                    success = any(future_prices < current_price)
                else:
                    continue
                
                accuracy[signal_type][horizon]['total'] += 1
                if success:
                    accuracy[signal_type][horizon]['success'] += 1

        # Store results
        self.accuracy_results = {
            'Buy': {h: (accuracy['Buy'][h]['success'] / accuracy['Buy'][h]['total'] * 100 
                      if accuracy['Buy'][h]['total'] > 0 else 0) 
                      for h in horizons},
            'Sell': {h: (accuracy['Sell'][h]['success'] / accuracy['Sell'][h]['total'] * 100 
                       if accuracy['Sell'][h]['total'] > 0 else 0) 
                       for h in horizons}
        }
        return self.accuracy_results
# ======================
# Visualization
# ======================
def plot_apple_analysis(data, analyzer, accuracy_results):
    plt.figure(figsize=(24, 14))
    
    # Price Chart
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(data['Close'], label='Price', color='black', linewidth=1)
    plt.title('Apple Stock Analysis with SMC Signals and Accuracy Metrics')
    
    # Plot signals with success/failure markers
    signals = data[data['Signal'].notnull()]
    for idx, row in signals.iterrows():
        pos = data.index.get_loc(idx)
        signal_type = row['Signal']
        success_5d = False
        
        if pos + 5 < len(data):
            future_prices = data['Close'].iloc[pos+1:pos+6]
            if signal_type == 'Buy':
                success_5d = any(future_prices > row['Close'])
            elif signal_type == 'Sell':
                success_5d = any(future_prices < row['Close'])
        
        color = 'green' if success_5d else 'red'
        marker = '^' if signal_type == 'Buy' else 'v'
        plt.plot(idx, row['Close'], 
                marker=marker, 
                markersize=10,
                color=color,
                markeredgecolor='black',
                linestyle='None')

    # Market Structure Elements
    for ms in analyzer.market_structure:
        if ms[0] == 'BOS':
            plt.plot(ms[2], data.loc[ms[2]]['Close'], 
                    marker='*' if ms[1] == 'bullish' else 'X',
                    markersize=12, 
                    color='lime' if ms[1] == 'bullish' else 'magenta')

    # Accuracy Annotation
    accuracy_text = (
        f"5-Day Signal Accuracy:\n"
        f"Buy: {accuracy_results['Buy'][5]:.1f}%\n"
        f"Sell: {accuracy_results['Sell'][5]:.1f}%"
    )
    plt.annotate(accuracy_text, xy=(0.02, 0.95),
                xycoords='axes fraction',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Volume Chart
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    plt.bar(data.index, data['Volume'], color=np.where(data['Close'] > data['Open'], 'g', 'r'), alpha=0.7)
    plt.title('Volume Profile')
    plt.ylabel('Volume')

    plt.tight_layout()
    plt.show()
# ======================
# Usage Example
# ======================
if __name__ == "__main__":
    system = Apple_SMC_TradingSystem()
    
    # Fetch Apple stock data
    data = system.fetch_data('AAPL', 
                            start_date='2024-01-01', 
                            end_date='2025-01-26'
                            # end_date=datetime.now().strftime('%Y-%m-%d')
                            )
    
    # Perform SMC analysis
    analyzed_data = system.full_analysis()
    
    # Generate trading signals
    signal_data = system.generate_signals()

    # Calculate accuracy
    accuracy = system.calculate_accuracy()
    # Visualize analysis
     # Print results
    print("\nSignal Accuracy Report:")
    print(f"Buy Signals (1-day): {accuracy['Buy'][1]:.1f}%")
    print(f"Buy Signals (3-day): {accuracy['Buy'][3]:.1f}%")
    print(f"Buy Signals (5-day): {accuracy['Buy'][5]:.1f}%")
    print(f"\nSell Signals (1-day): {accuracy['Sell'][1]:.1f}%")
    print(f"Sell Signals (3-day): {accuracy['Sell'][3]:.1f}%")
    print(f"Sell Signals (5-day): {accuracy['Sell'][5]:.1f}%")
    # Display key metrics
    print("Apple Stock SMC Analysis Report:")  # Updated title
    print(f"Market Structure Events: {len(system.analyzer.market_structure)}")
    print(f"Order Blocks Identified: {len(system.analyzer.order_blocks)}")
    print(f"Liquidity Zones Found: {len(system.analyzer.liquidity_zones['swing_highs'])} Swing Highs, "
          f"{len(system.analyzer.liquidity_zones['swing_lows'])} Swing Lows")
    print(f"Fair Value Gaps Detected: {len(system.analyzer.imbalances)}")
    print(f"Wyckoff Phases Identified: {len(system.analyzer.wyckoff_phases)}")
    
    plot_apple_analysis(signal_data, system.analyzer, accuracy)
    