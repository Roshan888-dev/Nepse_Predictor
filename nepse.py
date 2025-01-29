import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ======================
# Data Requirements for NEPSE
# ======================
"""
Required Data Structure (CSV/Excel):
- Date (YYYY-MM-DD)
- Open
- High
- Low
- Close
- Volume
- Symbol (Stock Ticker)
- VWAP (Volume Weighted Average Price) - Optional but recommended
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
    
    # Market Structure Analysis
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
    
    # Order Block Detection
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
    
    # Liquidity Zone Identification
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
    
    # Fair Value Gap (Imbalance) Detection
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
    
    # Wyckoff Phase Analysis
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
# NEPSE Trading System
# ======================
class NEPSE_SMC_TradingSystem:
    def __init__(self):
        self.data = None
        self.analyzer = SMC_Analyzer()
    
    def load_data(self, filepath):
        self.data = pd.read_csv(filepath, parse_dates=['Date'])
        self.data.set_index('Date', inplace=True)
        return self.data
    
    def full_analysis(self):
        self.analyzer.analyze_market_structure(self.data)
        self.analyzer.identify_order_blocks(self.data)
        self.analyzer.find_liquidity_zones(self.data)
        self.analyzer.detect_imbalances(self.data)
        self.analyzer.analyze_wyckoff(self.data)
        return self.data
    
    def generate_signals(self):
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

# ======================
# Visualization
# ======================
def plot_nepse_analysis(data, analyzer):
    plt.figure(figsize=(20, 12))
    
    # Price and Volume
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(data['Close'], label='Price', color='black')
    plt.title('NEPSE SMC Analysis')
    
    # Market Structure
    for ms in analyzer.market_structure:
        if ms[0] == 'BOS':
            plt.plot(ms[2], data.loc[ms[2]]['Close'], 
                    marker='^' if ms[1] == 'bullish' else 'v',
                    markersize=10, color='green' if ms[1] == 'bullish' else 'red')
    
    # Order Blocks
    for ob in analyzer.order_blocks:
        plt.plot(ob[1], ob[2], 
                marker='o' if 'Bullish' in ob[0] else 's',
                color='green' if 'Bullish' in ob[0] else 'red')
    
    # Imbalances
    for imb in analyzer.imbalances:
        plt.axhspan(imb[2], imb[3], alpha=0.2, 
                   color='green' if 'Bullish' in imb[0] else 'red')
    
    # Volume Analysis
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    plt.bar(data.index, data['Volume'], color=np.where(data['Close'] > data['Open'], 'g', 'r'))
    plt.title('Volume Analysis')
    
    plt.tight_layout()
    plt.show()

# ======================
# Usage Example
# ======================
if __name__ == "__main__":
    system = NEPSE_SMC_TradingSystem()
    
    # Load NEPSE data (example format)
    # Replace with actual NEPSE data file
    data = system.load_data('nepse_data.csv')
    
    # Perform SMC analysis
    analyzed_data = system.full_analysis()
    
    # Generate trading signals
    signal_data = system.generate_signals()
    
    # Visualize analysis
    plot_nepse_analysis(signal_data, system.analyzer)
    
    # Display key metrics
    print("SMC Analysis Report:")
    print(f"Market Structure Events: {len(system.analyzer.market_structure)}")
    print(f"Order Blocks Identified: {len(system.analyzer.order_blocks)}")
    print(f"Liquidity Zones Found: {len(system.analyzer.liquidity_zones['swing_highs'])} Swing Highs, "
          f"{len(system.analyzer.liquidity_zones['swing_lows'])} Swing Lows")
    print(f"Fair Value Gaps Detected: {len(system.analyzer.imbalances)}")
    print(f"Wyckoff Phases Identified: {len(system.analyzer.wyckoff_phases)}")