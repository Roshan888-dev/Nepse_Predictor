import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
from matplotlib.dates import date2num, DateFormatter
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib.patches import Rectangle

# ======================
# Enhanced SMC Analyzer
# ======================
class SMC_Analyzer:
    def __init__(self):
        self.bos_choch = []       # List of BOS/CHOCH events as lines between two candles
        self.liquidity_sweeps = []
        self.false_breakouts = []
        self.imbalances = []      # FVG zones as rectangles
        self.trades = []          # Planned trades with entry, stop, and take profit

    def detect_bos_choch(self, df):
        """
        Detect BOS and CHOCH events and store them as a line between the prior candle and current candle.
        Each event is stored as:
           (event_type, direction, start_time, end_time, start_value, end_value)
        """
        highs = df['High'].values
        lows = df['Low'].values
        dates = df.index
        events = []
        
        for i in range(2, len(df) - 2):
            # --- BOS Detection ---
            # Bullish BOS: a rising sequence (adjust logic as needed)
            if highs[i] > highs[i - 1] > highs[i - 2]:
                events.append(('BOS', 'bullish',
                               dates[i - 1], dates[i],
                               highs[i - 1], highs[i]))
            # Bearish BOS
            elif lows[i] < lows[i - 1] < lows[i - 2]:
                events.append(('BOS', 'bearish',
                               dates[i - 1], dates[i],
                               lows[i - 1], lows[i]))
            
            # --- CHOCH Detection ---
            # Bearish CHOCH: example condition (change from bullish to bearish)
            if (highs[i] < highs[i - 1] and highs[i - 1] > highs[i - 2] and
                lows[i] > lows[i - 1] and lows[i - 1] < lows[i - 2]):
                events.append(('CHOCH', 'bearish',
                               dates[i - 1], dates[i],
                               highs[i - 1], highs[i]))
            # Bullish CHOCH
            elif (lows[i] > lows[i - 1] and lows[i - 1] < lows[i - 2] and
                  highs[i] < highs[i - 1] and highs[i - 1] > highs[i]):
                events.append(('CHOCH', 'bullish',
                               dates[i - 1], dates[i],
                               lows[i - 1], lows[i]))
        
        self.bos_choch = events
        return events

    def detect_liquidity_sweeps(self, df):
        """
        Detect liquidity sweeps.
        Each event is stored as:
           (sweep_type, event_time, price)
        """
        highs = df['High'].values
        lows = df['Low'].values
        dates = df.index
        sweeps = []
        for i in range(2, len(df) - 2):
            # Bullish liquidity sweep
            if (highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and
                    lows[i] > lows[i - 1] and lows[i] > lows[i - 2]):
                sweeps.append(('Bullish Sweep', dates[i], highs[i]))
            # Bearish liquidity sweep
            if (lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and
                    highs[i] < highs[i - 1] and highs[i] < highs[i - 2]):
                sweeps.append(('Bearish Sweep', dates[i], lows[i]))
        self.liquidity_sweeps = sweeps
        return sweeps

    def detect_false_breakouts(self, df):
        """
        Detect false breakouts.
        Each event is stored as:
           (breakout_type, event_time, price)
        """
        highs = df['High'].values
        lows = df['Low'].values
        dates = df.index
        false_breakouts = []
        for i in range(2, len(df) - 2):
            # Bullish false breakout
            if (highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and
                    df['Close'].iloc[i] < df['Open'].iloc[i]):
                false_breakouts.append(('False Bullish Breakout', dates[i], highs[i]))
            # Bearish false breakout
            if (lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and
                    df['Close'].iloc[i] > df['Open'].iloc[i]):
                false_breakouts.append(('False Bearish Breakout', dates[i], lows[i]))
        self.false_breakouts = false_breakouts
        return false_breakouts

    def detect_imbalances(self, df):
        """
        Detect imbalances (Fair Value Gaps, FVG).
        Each imbalance is stored as:
           (zone_type, start_time, end_time, lower_bound, upper_bound)
        """
        imbalances = []
        dates = df.index
        for i in range(0, len(df) - 1):
            current_low = df['Low'].iloc[i]
            current_high = df['High'].iloc[i]
            next_low = df['Low'].iloc[i + 1]
            next_high = df['High'].iloc[i + 1]
            
            # Bullish imbalance (FVG): gap above current high and below next low
            if current_high < next_low:
                imbalances.append(('Bullish FVG',
                                   dates[i], dates[i + 1],
                                   current_high, next_low))
            # Bearish imbalance (FVG): gap below current low and above next high
            if current_low > next_high:
                imbalances.append(('Bearish FVG',
                                   dates[i], dates[i + 1],
                                   next_high, current_low))
        self.imbalances = imbalances
        return imbalances

    def plan_entries_exits(self, df, reward_ratio=2):
        """
        For each BOS/CHOCH event, plan an entry trade with stop loss and take profit.
        This example uses the following rules:
        
         - For a bullish event:
              * Entry = end candle's high (from BOS/CHOCH line)
              * Stop Loss = minimum(low of start candle, low of end candle)
              * Risk = entry - stop loss
              * Take Profit = entry + risk * reward_ratio
         - For a bearish event:
              * Entry = end candle's low (from BOS/CHOCH line)
              * Stop Loss = maximum(high of start candle, high of end candle)
              * Risk = stop loss - entry
              * Take Profit = entry - risk * reward_ratio
        """
        trades = []
        df = df.sort_index()
        for event in self.bos_choch:
            event_type, direction, start_time, end_time, start_value, end_value = event
            try:
                candle_start = df.loc[start_time]
                candle_end = df.loc[end_time]
            except KeyError:
                continue  # Skip events if candle timestamps do not exactly match

            if direction == 'bullish':
                entry_price = end_value
                stop_loss = min(candle_start['Low'], candle_end['Low'])
                risk = entry_price - stop_loss
                take_profit = entry_price + risk * reward_ratio
            else:  # bearish
                entry_price = end_value
                stop_loss = max(candle_start['High'], candle_end['High'])
                risk = stop_loss - entry_price
                take_profit = entry_price - risk * reward_ratio

            trades.append({
                'event_type': event_type,
                'direction': direction,
                'entry_time': end_time,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk': risk,
                'reward_ratio': reward_ratio
            })
        self.trades = trades
        return trades

# ======================
# Chart Plotting Function
# ======================
def plot_smc_chart(df, analyzer, ticker):
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Prepare candlestick data
    data_ohlc = df.reset_index()
    # Make sure the date column is named 'Date'
    if 'Date' not in data_ohlc.columns:
        data_ohlc.rename(columns={data_ohlc.columns[0]: 'Date'}, inplace=True)
    data_ohlc['Date'] = data_ohlc['Date'].map(date2num)
    ohlc = data_ohlc[['Date', 'Open', 'High', 'Low', 'Close']].values
    candlestick_ohlc(ax, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.8)

    # Plot BOS/CHOCH events as lines connecting two candle points
    for event in analyzer.bos_choch:
        event_type, direction, start_time, end_time, start_value, end_value = event
        start_num = date2num(start_time)
        end_num = date2num(end_time)
        color = 'blue' if direction == 'bullish' else 'red'
        label = f"{event_type} ({direction})"
        ax.plot([start_num, end_num], [start_value, end_value],
                color=color, linewidth=2, label=label)

    # Plot Liquidity Sweeps as markers
    for sweep in analyzer.liquidity_sweeps:
        label, event_time, price = sweep
        ax.plot(date2num(event_time), price, 'o', 
                color=('lime' if 'Bullish' in label else 'magenta'),
                markersize=8, label=label)

    # Plot False Breakouts as markers
    for breakout in analyzer.false_breakouts:
        label, event_time, price = breakout
        ax.plot(date2num(event_time), price, 'x', 
                color=('orange' if 'Bullish' in label else 'purple'),
                markersize=10, label=label)

    # Plot Imbalances (FVG zones) as rectangles
    for imb in analyzer.imbalances:
        zone_type, start_time, end_time, lower_bound, upper_bound = imb
        x_start = date2num(start_time)
        x_end = date2num(end_time)
        width = x_end - x_start
        height = upper_bound - lower_bound
        color = 'yellow' if 'Bullish' in zone_type else 'pink'
        rect = Rectangle((x_start, lower_bound), width, height, color=color, alpha=0.3, label=zone_type)
        ax.add_patch(rect)

    # Plot planned trades: entry, stop loss, and take profit
    for trade in analyzer.trades:
        entry_time = trade['entry_time']
        entry_price = trade['entry_price']
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']
        direction = trade['direction']
        entry_num = date2num(entry_time)
        # Plot entry marker
        marker = 'o' if direction == 'bullish' else 'x'
        color = 'green' if direction == 'bullish' else 'red'
        ax.plot(entry_num, entry_price, marker, color=color, markersize=12, label=f"Entry ({direction})")
        # Draw horizontal lines for stop loss and take profit
        ax.hlines(stop_loss, entry_num - 0.5, entry_num + 0.5, colors='black', linestyles='dashed', label="Stop Loss")
        ax.hlines(take_profit, entry_num - 0.5, entry_num + 0.5, colors='gray', linestyles='dashed', label="Take Profit")
        # Annotate risk (distance between entry and stop loss)
        ax.annotate(f"R: {trade['risk']:.2f}", xy=(entry_num, entry_price), 
                    xytext=(entry_num, entry_price * (1.01 if direction=='bullish' else 0.99)),
                    arrowprops=dict(facecolor=color, arrowstyle="->"))
    
    # Formatting
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.set_title(f'SMC Chart for {ticker} with SMC Prediction & Trade Entries/Exits')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    
    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    # List of tickers you want to plot
    tickers = ['NVDA', 'AAPL', 'MSFT']  # replace with your stocks
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {ticker}.")
            continue
        
        # Initialize and run the SMC Analyzer for the ticker
        analyzer = SMC_Analyzer()
        analyzer.detect_bos_choch(data)
        analyzer.detect_liquidity_sweeps(data)
        analyzer.detect_false_breakouts(data)
        analyzer.detect_imbalances(data)
        analyzer.plan_entries_exits(data, reward_ratio=2)
        
        # Plot the chart
        plot_smc_chart(data, analyzer, ticker)
