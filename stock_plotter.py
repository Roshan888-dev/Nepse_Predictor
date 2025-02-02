import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

class SmartTradingSystem:
    def __init__(self, ticker, risk_reward_ratio=3):
        self.ticker = ticker
        self.data = None
        self.signals = pd.DataFrame()
        self.risk_reward = risk_reward_ratio  # 1:3 by default
    
    def fetch_data(self, start_date, end_date):
        self.data = yf.download(self.ticker, start=start_date, end=end_date)
        self.data['VWAP'] = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        return self.data
    
    def calculate_support_resistance(self, window=20):
        self.data['Support'] = self.data['Low'].rolling(window=window).min()
        self.data['Resistance'] = self.data['High'].rolling(window=window).max()
        return self.data
    
    def generate_signals(self):
        # Simple momentum-based signal generation (customize this with your strategy)
        self.data['Signal'] = np.where(
            self.data['Close'] > self.data['VWAP'], 1, 
            np.where(self.data['Close'] < self.data['VWAP'], -1, 0)
        )
        
        # Calculate trade parameters
        self.data['Entry_Price'] = np.where(
            self.data['Signal'].diff() != 0, 
            self.data['Close'], 
            np.nan
        )
        
        # Calculate stop loss and take profit levels
        self.data['Stop_Loss'] = np.where(
            self.data['Signal'] == 1,
            self.data['Low'].rolling(5).min(),
            self.data['High'].rolling(5).max()
        )
        
        self.data['Take_Profit'] = np.where(
            self.data['Signal'] == 1,
            self.data['Entry_Price'] + (self.data['Entry_Price'] - self.data['Stop_Loss']) * self.risk_reward,
            self.data['Entry_Price'] - (self.data['Stop_Loss'] - self.data['Entry_Price']) * self.risk_reward
        )
        
        return self.data
    
    def plot_trading_chart(self):
        plt.figure(figsize=(20, 12))
        
        # Price chart
        ax = plt.subplot(1, 1, 1)
        plt.plot(self.data['Close'], label='Price', color='#2E86C1', linewidth=2)
        plt.title(f'{self.ticker} Trading Chart with 1:{self.risk_reward} Risk/Reward Zones', fontsize=16, pad=20)
        
        # Plot support/resistance
        plt.plot(self.data['Support'], color='#27AE60', linestyle='--', alpha=0.7, label='Support')
        plt.plot(self.data['Resistance'], color='#E74C3C', linestyle='--', alpha=0.7, label='Resistance')
        
        # Plot signals and risk/reward zones
        for idx, row in self.data.iterrows():
            if not np.isnan(row['Entry_Price']):
                entry_color = '#2ECC71' if row['Signal'] == 1 else '#E74C3C'
                direction = 'Long' if row['Signal'] == 1 else 'Short'
                
                # Entry price
                plt.scatter(idx, row['Entry_Price'], color=entry_color, s=100, 
                           edgecolor='black', zorder=5, label=f'{direction} Entry')
                
                # Stop loss and take profit lines
                plt.hlines(y=row['Stop_Loss'], xmin=idx, xmax=self.data.index[-1], 
                          colors='#E74C3C', linestyles='dashed', alpha=0.7)
                plt.hlines(y=row['Take_Profit'], xmin=idx, xmax=self.data.index[-1], 
                          colors='#27AE60', linestyles='dashed', alpha=0.7)
                
                # Risk/reward zone shading
                if row['Signal'] == 1:
                    plt.fill_betweenx(y=[row['Stop_Loss'], row['Take_Profit']], 
                                     x=idx, x1=self.data.index[-1],
                                     color='#27AE60', alpha=0.1)
                else:
                    plt.fill_betweenx(y=[row['Take_Profit'], row['Stop_Loss']], 
                                     x=idx, x1=self.data.index[-1],
                                     color='#E74C3C', alpha=0.1)
                
                # Annotation box
                text = f"{direction} Entry: {row['Entry_Price']:.2f}\nStop: {row['Stop_Loss']:.2f}\nTarget: {row['Take_Profit']:.2f}\nR:R 1:{self.risk_reward}"
                plt.annotate(text, xy=(idx, row['Entry_Price']),
                            xytext=(10, -20), textcoords='offset points',
                            arrowprops=dict(arrowstyle="->", color='black'),
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Formatting
        plt.legend(loc='upper left')
        plt.ylabel('Price', fontsize=12)
        plt.grid(alpha=0.2)
        ax.set_facecolor('#F8F9F9')
        plt.gcf().autofmt_xdate()
        
        # Add risk/reward legend
        plt.annotate(f'Risk/Reward Ratio: 1:{self.risk_reward}\nGreen Zone: Reward Area\nRed Zone: Risk Area', 
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='white'))
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize system with desired risk/reward ratio
    trading_system = SmartTradingSystem(ticker='AAPL', risk_reward_ratio=3)
    
    # Fetch data (adjust date range as needed)
    data = trading_system.fetch_data(start_date='2023-01-01', end_date='2024-01-01')
    
    # Calculate technical levels
    trading_system.calculate_support_resistance()
    
    # Generate signals and trade parameters
    trading_system.generate_signals()
    
    # Plot interactive trading chart
    trading_system.plot_trading_chart()