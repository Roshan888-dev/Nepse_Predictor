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
    # Create a figure with two subplots
    fig = plt.figure(figsize=(24, 18))
    ax1 = plt.subplot(2, 1, 1)
    
    # Prepare data for candlestick chart
    data_ohlc = data.copy().reset_index()
    data_ohlc['Date'] = data_ohlc['Date'].map(date2num)
    ohlc = data_ohlc[['Date','Open','High','Low','Close']].values

    # Plot candlestick chart
    candlestick_ohlc(ax1, ohlc, width=0.6, colorup='g', colordown='r', alpha=0.8)
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.title('NVDA Stock Analysis with SMC Signals and Market Structure')

    # Custom legend tracker
    legend_elements = {}
    
    # Plot signals with enhanced markers
    signals = data[data['Signal'].notnull()]
    for idx, row in signals.iterrows():
        date_num = date2num(idx)
        signal_type = row['Signal']
        
        # Fix: apply any() only once depending on the signal type.
        if signal_type == 'Buy':
            success_5d = any(data['Close'].loc[idx:idx+pd.Timedelta(days=5)] > row['Close'])
        else:
            success_5d = any(data['Close'].loc[idx:idx+pd.Timedelta(days=5)] < row['Close'])
        
        marker_props = {
            'marker': '^' if signal_type == 'Buy' else 'v',
            'markersize': 12,
            'color': 'lime' if success_5d else 'red',
            'markeredgecolor': 'black',
            'zorder': 9
        }
        ax1.plot(date_num, row['Close'], **marker_props, linestyle='None')
        legend_label = f"{signal_type} Signal ({'Success' if success_5d else 'Fail'})"
        if legend_label not in legend_elements:
            legend_elements[legend_label] = marker_props

    # Plot Demand/Supply Zones with shading
    zone_alpha = 0.2
    for zone in analyzer.demand_zones:
        start = date2num(zone[1] - pd.Timedelta(days=2))
        end = date2num(zone[1] + pd.Timedelta(days=2))
        ax1.axhspan(zone[2]*0.995, zone[2]*1.005, start, end, 
                    facecolor='green', alpha=zone_alpha)
        if 'Demand Zone' not in legend_elements:
            legend_elements['Demand Zone'] = plt.Rectangle((0,0),1,1, fc='green', alpha=zone_alpha)

    for zone in analyzer.supply_zones:
        start = date2num(zone[1] - pd.Timedelta(days=2))
        end = date2num(zone[1] + pd.Timedelta(days=2))
        ax1.axhspan(zone[2]*0.995, zone[2]*1.005, start, end,
                    facecolor='red', alpha=zone_alpha)
        if 'Supply Zone' not in legend_elements:
            legend_elements['Supply Zone'] = plt.Rectangle((0,0),1,1, fc='red', alpha=zone_alpha)

    # Plot Order Blocks with connection lines
    for block in analyzer.order_blocks:
        date_num = date2num(block[1])
        style = '^' if 'Bullish' in block[0] else 'v'
        color = 'darkgreen' if 'Bullish' in block[0] else 'darkred'
        ax1.plot(date_num, block[2], style, markersize=10, color=color,
                 markeredgecolor='black', zorder=8)
        # Draw horizontal line to current price
        current_price = data['Close'].iloc[-1]
        ax1.plot([date_num, date2num(data.index[-1])], [block[2], current_price],
                 color=color, alpha=0.3, linestyle='--')
        legend_label = f"{block[0]} Trendline"
        if legend_label not in legend_elements:
            legend_elements[legend_label] = plt.Line2D([0],[0], color=color, linestyle='--')

    # Enhanced Accuracy Annotation
    accuracy_text = (
        "Signal Accuracy:\n"
        "Buy Signals:\n"
        f"  1-Day: {accuracy_results['Buy'][1]:.1f}%\n"
        f"  3-Day: {accuracy_results['Buy'][3]:.1f}%\n"
        f"  5-Day: {accuracy_results['Buy'][5]:.1f}%\n"
        "\nSell Signals:\n"
        f"  1-Day: {accuracy_results['Sell'][1]:.1f}%\n"
        f"  3-Day: {accuracy_results['Sell'][3]:.1f}%\n"
        f"  5-Day: {accuracy_results['Sell'][5]:.1f}%"
    )
    ax1.annotate(accuracy_text, xy=(0.75, 0.95), xycoords='axes fraction',
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                 verticalalignment='top')

    # Market Structure Statistics
    stats_text = (
        "Market Structure:\n"
        f"Order Blocks: {len(analyzer.order_blocks)}\n"
        f"Demand/Supply Zones: {len(analyzer.demand_zones)}/{len(analyzer.supply_zones)}\n"
        f"Wyckoff Phases: {len(analyzer.wyckoff_phases)}\n"
        f"Fair Value Gaps: {len(analyzer.imbalances)}"
    )
    ax1.annotate(stats_text, xy=(0.02, 0.15), xycoords='axes fraction',
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Create unified legend
    handles = [plt.Line2D([0], [0], **props) if isinstance(props, dict) 
               else props for label, props in legend_elements.items()]
    labels = list(legend_elements.keys())
    ax1.legend(handles, labels, loc='upper left', fontsize=10)

    # Volume Chart with SMA
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    volume_colors = np.where(data['Close'] > data['Open'], 'g', 'r')
    ax2.bar(data_ohlc['Date'], data['Volume'], color=volume_colors, width=0.6, alpha=0.7)
    
    # Add 20-period Volume SMA
    volume_sma = data['Volume'].rolling(20).mean()
    ax2.plot(data_ohlc['Date'], volume_sma, color='purple', linewidth=2, label='20-SMA')
    ax2.legend()
    
    plt.ylabel('Volume')
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.title('Volume with 20-period SMA')
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
    print(f"Demand Zones Identified: {len(system.analyzer.demand_zones)}")
    print(f"Supply Zones Identified: {len(system.analyzer.supply_zones)}")
    print(f"MMC Levels Identified: {len(system.analyzer.mmc_levels)}")
    
    plot_apple_analysis(signal_data, system.analyzer, accuracy)