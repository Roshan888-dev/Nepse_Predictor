import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf  # For fetching data from Yahoo Finance
from matplotlib.dates import date2num, DateFormatter
from mplfinance.original_flavor import candlestick_ohlc  # For candlestick charting

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

    def detect_elliott_waves(self, df, swing_period=5):
        waves = []
        # Get swing highs and lows and reset their indices so we can use positional indexing safely.
        swing_highs = df['High'][(df['High'] > df['High'].shift(1)) & 
                                (df['High'] > df['High'].shift(-1))].reset_index(drop=True)
        swing_lows = df['Low'][(df['Low'] < df['Low'].shift(1)) & 
                                (df['Low'] < df['Low'].shift(-1))].reset_index(drop=True)
        
        # For impulse waves we need:
        # - For swing_highs: indices i to i+4 (so i+4 must be < len(swing_highs))
        # - For swing_lows: indices i to i+5 (so i+5 must be < len(swing_lows))
        max_index = min(len(swing_highs) - 4, len(swing_lows) - 5)
        
        for i in range(0, max_index):
            wave1 = (swing_lows.iloc[i] < swing_lows.iloc[i+1] < 
                    swing_lows.iloc[i+2] < swing_lows.iloc[i+3] < 
                    swing_lows.iloc[i+4])
            wave2 = swing_highs.iloc[i] > swing_highs.iloc[i+1]
            wave3 = swing_lows.iloc[i+2] < swing_lows.iloc[i+3]
            wave4 = swing_highs.iloc[i+3] > swing_highs.iloc[i+4]
            wave5 = swing_lows.iloc[i+4] < swing_lows.iloc[i+5]
            
            if wave1 and wave2 and wave3 and wave4 and wave5:
                # Use the original DataFrame's index for the wave points.
                # Since the swing series have been reset, you might need to map these positions
                # back to the original index if required.
                waves.append(('Impulse', 'bullish', 
                            (df.index[i], swing_lows.iloc[i]),
                            (df.index[i+1], swing_highs.iloc[i+1]),
                            (df.index[i+2], swing_lows.iloc[i+2]),
                            (df.index[i+3], swing_highs.iloc[i+3]),
                            (df.index[i+4], swing_lows.iloc[i+4])))
        
        # For corrective waves we need indices i to i+3 for swing highs and i to i+2 for swing lows.
        max_index_corr = min(len(swing_highs) - 3, len(swing_lows) - 2)
        
        for i in range(0, max_index_corr):
            waveA = swing_highs.iloc[i] > swing_highs.iloc[i+1]
            waveB = swing_lows.iloc[i+1] < swing_lows.iloc[i+2]
            waveC = swing_highs.iloc[i+2] > swing_highs.iloc[i+3]
            
            if waveA and waveB and waveC:
                waves.append(('Corrective', 'bearish',
                            (df.index[i], swing_highs.iloc[i]),
                            (df.index[i+1], swing_lows.iloc[i+1]),
                            (df.index[i+2], swing_highs.iloc[i+2])))
        
        self.elliott_waves = waves
        return waves


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
        self.analyzer.detect_elliott_waves(self.data)
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

    # ======================
    # Enhanced SMC Concepts
    # ======================
    
    # Highlight Major Order Blocks with larger markers
    for block in analyzer.order_blocks:
        date_num = date2num(block[1])
        if 'Bullish' in block[0]:
            ax1.plot(date_num, block[2], '^', markersize=12, color='lime', alpha=0.7, 
                     markeredgecolor='black', zorder=9, label='Major Bullish OB')
        else:
            ax1.plot(date_num, block[2], 'v', markersize=12, color='magenta', alpha=0.7,
                     markeredgecolor='black', zorder=9, label='Major Bearish OB')
    
    # Draw Elliott Waves
    wave_colors = {'Impulse': 'blue', 'Corrective': 'orange'}
    for wave in analyzer.elliott_waves:
        wave_type = wave[0]
        color = wave_colors.get(wave_type, 'gray')
        points = wave[2:]  # Get wave points
        
        # Draw connecting lines
        for i in range(len(points)-1):
            start_date = date2num(points[i][0])
            end_date = date2num(points[i+1][0])
            ax1.plot([start_date, end_date], [points[i][1], points[i+1][1]],
                    color=color, linewidth=2, linestyle='--', alpha=0.7)
        
        # Add wave labels
        label_pos = date2num(points[0][0]) + 0.2*(date2num(points[-1][0]) - date2num(points[0][0]))
        price_pos = min(p[1] for p in points) if wave_type == 'Impulse' else max(p[1] for p in points)
        label = f"{wave_type} Wave\n({wave[1]})"
        ax1.text(label_pos, price_pos, label, color=color, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Highlight Key SMC Elements
    # Draw rectangles around major FVGs
    for imb in analyzer.imbalances:
        if abs(imb[3] - imb[2]) > data['Close'].mean() * 0.03:  # Only major gaps
            date_num = date2num(imb[1])
            rect = plt.Rectangle((date_num-0.4, imb[2]), 0.8, imb[3]-imb[2],
                                color='yellow' if 'Bullish' in imb[0] else 'purple',
                                alpha=0.3, zorder=1)
            ax1.add_patch(rect)
    
    # Annotate Wyckoff Phases
    for phase in analyzer.wyckoff_phases:
        date_num = date2num(phase[1])
        price = data.loc[phase[1]]['Close']
        ax1.annotate(phase[0], (date_num, price),
                    xytext=(0, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color='gray'),
                    fontsize=9, color='darkblue')
    
    # Draw support and resistance lines for the current trend.
    # For this example, we use the last 20 candles to define the current trend.
    N = 20
    if len(data) >= N:
        recent_data = data.iloc[-N:]
        support = recent_data['Low'].min()
        resistance = recent_data['High'].max()
        min_date = date2num(data.index[0])
        max_date = date2num(data.index[-1])
        ax1.hlines(y=support, xmin=min_date, xmax=max_date, colors='blue', linestyles='--', linewidth=1.5, label='Support')
        ax1.hlines(y=resistance, xmin=min_date, xmax=max_date, colors='red', linestyles='--', linewidth=1.5, label='Resistance')
        # Add text labels for support and resistance
        ax1.text(max_date, support, f' Support: {support:.2f}', verticalalignment='bottom', color='blue', fontsize=10)
        ax1.text(max_date, resistance, f' Resistance: {resistance:.2f}', verticalalignment='top', color='red', fontsize=10)
    
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
