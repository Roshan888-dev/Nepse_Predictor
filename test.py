import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from matplotlib.dates import date2num, DateFormatter
from mplfinance.original_flavor import candlestick_ohlc

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
        self.elliott_waves = []
        self.demand_zones = []
        self.supply_zones = []
        self.mmc_levels = []

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

    # Fair Value Gap Detection
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

    # Detect Demand and Supply Zones
    def detect_demand_supply_zones(self, df):
        demand_zones = []
        supply_zones = []
        
        for i in range(1, len(df)-1):
            # Demand Zone: Strong bullish candle after a bearish move
            if (df['Close'].iloc[i] > df['Open'].iloc[i] and
                df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and
                df['Volume'].iloc[i] > df['Volume'].iloc[i-1]):
                demand_zones.append(('Demand Zone', df.index[i], df['Low'].iloc[i]))
            
            # Supply Zone: Strong bearish candle after a bullish move
            if (df['Close'].iloc[i] < df['Open'].iloc[i] and
                df['Close'].iloc[i-1] > df['Open'].iloc[i-1] and
                df['Volume'].iloc[i] > df['Volume'].iloc[i-1]):
                supply_zones.append(('Supply Zone', df.index[i], df['High'].iloc[i]))
        
        self.demand_zones = demand_zones
        self.supply_zones = supply_zones
        return demand_zones, supply_zones

    # Market Maker Concept (MMC) Levels
    def detect_mmc_levels(self, df):
        mmc_levels = []
        for i in range(1, len(df)-1):
            # Key Levels: Previous Highs and Lows
            if df['High'].iloc[i] == df['High'].iloc[i-1]:
                mmc_levels.append(('MMC Resistance', df.index[i], df['High'].iloc[i]))
            if df['Low'].iloc[i] == df['Low'].iloc[i-1]:
                mmc_levels.append(('MMC Support', df.index[i], df['Low'].iloc[i]))
        
        self.mmc_levels = mmc_levels
        return mmc_levels

# ======================
# Apple Trading System
# ======================
class Apple_SMC_TradingSystem:
    def __init__(self):
        self.data = None
        self.analyzer = SMC_Analyzer()
    
    def fetch_data(self, ticker, start_date, end_date):
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
        self.analyzer.detect_demand_supply_zones(self.data)
        self.analyzer.detect_mmc_levels(self.data)
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
            
            # Check Demand and Supply Zones
            for zone in self.analyzer.demand_zones:
                if abs(price - zone[2]) < price*0.005:
                    signal = 'Buy'
            for zone in self.analyzer.supply_zones:
                if abs(price - zone[2]) < price*0.005:
                    signal = 'Sell'
            
            # Check MMC Levels
            for level in self.analyzer.mmc_levels:
                if abs(price - level[2]) < price*0.005:
                    if level[0] == 'MMC Support':
                        signal = 'Buy'
                    elif level[0] == 'MMC Resistance':
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
    # Create a figure with two subplots: one for the candlestick chart and one for volume
    fig = plt.figure(figsize=(24, 18))
    ax1 = plt.subplot(2, 1, 1)
    
    # Prepare data for candlestick chart
    data_ohlc = data.copy()
    data_ohlc.reset_index(inplace=True)  # Move the date index into a column
    data_ohlc['Date'] = data_ohlc['Date'].map(date2num)  # Convert dates to numeric format
    ohlc = data_ohlc[['Date','Open','High','Low','Close']].values

    # Plot the candlestick chart
    candlestick_ohlc(ax1, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.8)
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.title('NVDA Stock Analysis with SMC Signals, Elliott Waves, and Accuracy Metrics (Candlestick)')
    
    # Plot signals with success/failure markers on the candlestick chart
    signals = data[data['Signal'].notnull()]
    for idx, row in signals.iterrows():
        date_num = date2num(idx)
        signal_type = row['Signal']
        success_5d = False
        pos = data.index.get_loc(idx)
        if pos + 5 < len(data):
            future_prices = data['Close'].iloc[pos+1:pos+6]
            if signal_type == 'Buy':
                success_5d = any(future_prices > row['Close'])
            elif signal_type == 'Sell':
                success_5d = any(future_prices < row['Close'])
        marker_color = 'green' if success_5d else 'red'
        marker_style = '^' if signal_type == 'Buy' else 'v'
        ax1.plot(date_num, row['Close'], marker=marker_style, markersize=10,
                 color=marker_color, markeredgecolor='black', linestyle='None',
                 label=f"{signal_type} Signal")
    
    # Remove duplicate legend entries
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())

    # Highlight Major Order Blocks with larger markers
    for block in analyzer.order_blocks:
        date_num = date2num(block[1])
        if 'Bullish' in block[0]:
            ax1.plot(date_num, block[2], '^', markersize=12, color='lime', alpha=0.7, 
                     markeredgecolor='black', zorder=9, label='Major Bullish OB')
        else:
            ax1.plot(date_num, block[2], 'v', markersize=12, color='magenta', alpha=0.7,
                     markeredgecolor='black', zorder=9, label='Major Bearish OB')
    
    # Draw Demand and Supply Zones
    for zone in analyzer.demand_zones:
        date_num = date2num(zone[1])
        ax1.axhline(y=zone[2], color='green', linestyle='--', alpha=0.5, label='Demand Zone')
    for zone in analyzer.supply_zones:
        date_num = date2num(zone[1])
        ax1.axhline(y=zone[2], color='red', linestyle='--', alpha=0.5, label='Supply Zone')
    
    # Draw MMC Levels
    for level in analyzer.mmc_levels:
        date_num = date2num(level[1])
        if level[0] == 'MMC Support':
            ax1.axhline(y=level[2], color='blue', linestyle='--', alpha=0.5, label='MMC Support')
        elif level[0] == 'MMC Resistance':
            ax1.axhline(y=level[2], color='orange', linestyle='--', alpha=0.5, label='MMC Resistance')
    
    # Add an accuracy annotation inside the candlestick subplot
    accuracy_text = (
        f"5-Day Signal Accuracy:\n"
        f"Buy: {accuracy_results['Buy'][5]:.1f}%\n"
        f"Sell: {accuracy_results['Sell'][5]:.1f}%"
    )
    ax1.annotate(accuracy_text, xy=(0.02, 0.95),
                 xycoords='axes fraction',
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Volume Chart
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    volume_dates = [date2num(idx) for idx in data.index]
    colors = np.where(data['Close'] > data['Open'], 'g', 'r')
    ax2.bar(volume_dates, data['Volume'], color=colors, alpha=0.7, width=0.6)
    plt.title('Volume Profile')
    plt.ylabel('Volume')
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    
    plt.tight_layout()
    plt.show()

# ======================
# Usage Example
# ======================
if __name__ == "__main__":
    system = Apple_SMC_TradingSystem()
    
    # Fetch Apple stock data
    data = system.fetch_data('NVDA', 
                               start_date='2024-01-01', 
                               end_date='2025-01-26')
    
    # Perform SMC analysis
    analyzed_data = system.full_analysis()
    
    # Generate trading signals
    signal_data = system.generate_signals()

    # Calculate accuracy
    accuracy = system.calculate_accuracy()
    
    # Print results
    print("\nSignal Accuracy Report:")
    print(f"Buy Signals (1-day): {accuracy['Buy'][1]:.1f}%")
    print(f"Buy Signals (3-day): {accuracy['Buy'][3]:.1f}%")
    print(f"Buy Signals (5-day): {accuracy['Buy'][5]:.1f}%")
    print(f"\nSell Signals (1-day): {accuracy['Sell'][1]:.1f}%")
    print(f"Sell Signals (3-day): {accuracy['Sell'][3]:.1f}%")
    print(f"Sell Signals (5-day): {accuracy['Sell'][5]:.1f}%")
    
    print("Apple Stock SMC Analysis Report:")
    print(f"Market Structure Events: {len(system.analyzer.market_structure)}")
    print(f"Order Blocks Identified: {len(system.analyzer.order_blocks)}")
    print(f"Liquidity Zones Found: {len(system.analyzer.liquidity_zones['swing_highs'])} Swing Highs, "
          f"{len(system.analyzer.liquidity_zones['swing_lows'])} Swing Lows")
    print(f"Fair Value Gaps Detected: {len(system.analyzer.imbalances)}")
    print(f"Wyckoff Phases Identified: {len(system.analyzer.wyckoff_phases)}")
    
    plot_apple_analysis(signal_data, system.analyzer, accuracy)
