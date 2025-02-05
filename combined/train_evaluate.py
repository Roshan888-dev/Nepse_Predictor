import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from yfinance import Ticker, download  # Add missing import

class SMCTrainer:
    def __init__(self, model, feature_engineer):
        self.model = model
        self.feature_engineer = feature_engineer

    def train(self, stock_list, start_date, end_date, epochs=50, batch_size=64, validation_split=0.2):
        all_features = []
        all_labels = []
        
        for ticker in stock_list:
            # Get historical data
            raw_data = Ticker(ticker).history(start=start_date, end=end_date)
            
            # Calculate features and labels
            features = self.feature_engineer.calculate_features(raw_data)
            labels = self._generate_labels(raw_data)
            
            # Align lengths
            min_length = min(len(features), len(labels))
            features = features[:min_length]
            labels = labels[:min_length]
            
            # Create sequences
            seq_features, seq_labels = self._create_sequences(features, labels)
            all_features.append(seq_features)
            all_labels.append(seq_labels)
        
        # Rest of training code remains the same
        X = np.concatenate(all_features)
        y = np.concatenate(all_labels)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks
        )
        
        return history

    def _generate_labels(self, data):
        """Generate target labels based on 3-day future returns"""
        labels = []
        future_returns = data['Close'].pct_change(3).shift(-3)
        
        for ret in future_returns:
            if ret > 0.02:  # 2% gain threshold
                labels.append(0)  # Buy
            elif ret < -0.02:  # 2% loss threshold
                labels.append(1)  # Sell
            else:
                labels.append(2)  # Hold
        
        return np.array(labels)[:len(data)-3]

    def _create_sequences(self, features, labels, window=30):
        """Create time-series sequences"""
        sequences = []
        seq_labels = []
        for i in range(len(features)-window):
            sequences.append(features[i:i+window])
            seq_labels.append(labels[i+window])
        return np.array(sequences), np.array(seq_labels)
    
    def analyze_market(self, universe='sp500', year_back=1, batch_size=50):
        """Analyze the entire market universe."""
        tickers = self._get_universe_tickers(universe)
        analysis_results = []
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            for ticker in batch_tickers:
                try:
                    data = download(ticker, period=f'{year_back}y')
                    if len(data) < 100:
                        continue
                        
                    features = self.feature_engineer.calculate_features(data)
                    signals = self.generate_signals(data)
                    positions = self.calculate_positions(data, signals)
                    
                    self.plot_analysis(ticker, data, signals, positions)
                    
                    analysis_results.append({
                        'ticker': ticker,
                        'current_signal': signals.iloc[-1],
                        'return_1m': data['Close'].pct_change(21).iloc[-1],
                        'volatility': data['Close'].pct_change().std() * np.sqrt(252)
                    })
                    
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
            
            plt.close('all')
        
        self.generate_market_report(analysis_results)
        return analysis_results

    def generate_signals(self, data):
        """Generate trading signals."""
        signals = []
        df = data.copy()
        
        df['ob_strength'] = self.feature_engineer._calculate_order_blocks(df)
        df['market_structure'] = self.feature_engineer._calculate_market_structure(df)
        df['liquidity_gradient'] = self.feature_engineer._calculate_liquidity_zones(df)
        df['imbalance_ratio'] = self.feature_engineer._calculate_imbalance_ratio(df)
        df['wyckoff_phase'] = self.feature_engineer._calculate_wyckoff_phases(df)

        for idx, row in df.iterrows():
            signal = 'Neutral'
            
            ob = row['ob_strength']
            ms = row['market_structure']
            liq = row['liquidity_gradient']
            imb = row['imbalance_ratio']
            wyk = row['wyckoff_phase']

            if any([ob > 0.8, ms in [1, 2], liq < -0.7, imb > 0.7, wyk == 1]):
                signal = 'Long'
            elif any([ob < -0.8, ms in [-1, -2], liq > 0.7, imb < -0.7, wyk == -1]):
                signal = 'Exit'
                
            signals.append(signal)
        
        return pd.Series(signals, index=df.index, name='signals')

    def calculate_positions(self, data, signals):
        """Calculate position sizes and risk parameters."""
        positions = pd.DataFrame(index=data.index)
        positions['signal'] = signals
        positions['position'] = 0.0
        positions['entry_price'] = np.nan
        positions['stop_loss'] = np.nan
        positions['take_profit'] = np.nan
        
        in_position = False
        for i in range(len(positions)):
            current_idx = positions.index[i]
            
            if not in_position and positions['signal'].iat[i] == 'Long':
                entry_price = data['Close'].iat[i]
                positions.at[current_idx, 'entry_price'] = entry_price
                positions.at[current_idx, 'stop_loss'] = entry_price * 0.95
                positions.at[current_idx, 'take_profit'] = entry_price * 1.10
                positions.at[current_idx, 'position'] = 0.01  # Risk 1% per trade
                in_position = True
            elif in_position:
                prev_idx = positions.index[i-1]
                
                if positions['signal'].iat[i] == 'Exit' or \
                data['Low'].iat[i] < positions.at[prev_idx, 'stop_loss'] or \
                data['High'].iat[i] > positions.at[prev_idx, 'take_profit']:
                    positions.at[current_idx, 'position'] = 0.0
                    in_position = False
                else:
                    positions.at[current_idx, 'position'] = positions.at[prev_idx, 'position']
                    positions.at[current_idx, 'entry_price'] = positions.at[prev_idx, 'entry_price']
                    positions.at[current_idx, 'stop_loss'] = positions.at[prev_idx, 'stop_loss'] * 1.01
                    positions.at[current_idx, 'take_profit'] = positions.at[prev_idx, 'take_profit'] * 1.005
                    
        return positions

    def plot_analysis(self, ticker, data, signals, positions):
        """Generate institutional-grade SMC analysis charts with detailed trade positioning."""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1])
        
        # ======================
        # Main Price Chart
        # ======================
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(data['Close'], label='Price', color='black', linewidth=1.5, zorder=5)
        
        # Plot key levels and zones
        self._plot_liquidity_zones(ax1, data)
        self._plot_order_blocks(ax1, data)
        self._plot_fair_value_gaps(ax1, data)
        
        # Plot trades with detailed positioning
        self._plot_trade_positions(ax1, data, positions)
        
        # Formatting
        ax1.set_title(f'{ticker} Institutional SMC Analysis', fontsize=20, pad=20)
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # ======================
        # Volume Profile
        # ======================
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        volume_colors = np.where(data['Close'] > data['Open'], 'forestgreen', 'firebrick')
        ax2.bar(data.index, data['Volume'], color=volume_colors, alpha=0.7, width=0.6)
        ax2.set_title('Volume Profile', fontsize=14, pad=15)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        
        # ======================
        # Market Structure
        # ======================
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        self._plot_market_structure(ax3, data)
        
        # ======================
        # Risk Management
        # ======================
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        self._plot_risk_parameters(ax4, positions)
        
        # Formatting
        plt.subplots_adjust(hspace=0.1)
        plt.xticks(rotation=45)
        
        # Save plot
        os.makedirs('analysis_charts', exist_ok=True)
        plt.savefig(f'analysis_charts/{ticker}_institutional_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_liquidity_zones(self, ax, data):
        """Plot liquidity zones on the price chart."""
        swing_highs = data['High'][(data['High'] > data['High'].shift(1)) & 
                                (data['High'] > data['High'].shift(-1))]
        swing_lows = data['Low'][(data['Low'] < data['Low'].shift(1)) & 
                                (data['Low'] < data['Low'].shift(-1))]
        
        for high in swing_highs:
            ax.axhline(high, color='red', linestyle='--', alpha=0.3, zorder=1)
        for low in swing_lows:
            ax.axhline(low, color='green', linestyle='--', alpha=0.3, zorder=1)

    def _plot_order_blocks(self, ax, data):
        """Plot order blocks on the price chart."""
        ob_bullish = data[data['ob_strength'] > 0.8]
        ob_bearish = data[data['ob_strength'] < -0.8]
        
        for idx, row in ob_bullish.iterrows():
            ax.axhspan(row['Low'], row['High'], color='green', alpha=0.1, zorder=2)
        for idx, row in ob_bearish.iterrows():
            ax.axhspan(row['Low'], row['High'], color='red', alpha=0.1, zorder=2)

    def _plot_fair_value_gaps(self, ax, data):
        """Plot fair value gaps on the price chart."""
        for i in range(1, len(data)-1):
            if data['Low'].iloc[i] > data['High'].iloc[i+1]:  # Bearish FVG
                ax.axhspan(data['High'].iloc[i+1], data['Low'].iloc[i], 
                        color='red', alpha=0.2, zorder=3)
            elif data['High'].iloc[i] < data['Low'].iloc[i+1]:  # Bullish FVG
                ax.axhspan(data['High'].iloc[i], data['Low'].iloc[i+1], 
                        color='green', alpha=0.2, zorder=3)

    def _plot_trade_positions(self, ax, data, positions):
        """Plot detailed trade positions with risk management."""
        in_position = False
        for i in range(len(positions)):
            if positions['position'].iloc[i] > 0 and not in_position:
                # Entry point
                entry_price = positions['entry_price'].iloc[i]
                stop_loss = positions['stop_loss'].iloc[i]
                take_profit = positions['take_profit'].iloc[i]
                
                # Plot entry
                ax.plot(positions.index[i], entry_price, 
                    marker='^', markersize=12, color='blue', zorder=10)
                
                # Plot SL and TP
                ax.hlines(stop_loss, positions.index[i-5], positions.index[i+5],
                        colors='red', linestyles='--', alpha=0.7)
                ax.hlines(take_profit, positions.index[i-5], positions.index[i+5],
                        colors='green', linestyles='--', alpha=0.7)
                
                # Fill between SL and TP
                ax.fill_between(positions.index[i-5:i+5], stop_loss, take_profit,
                            color='blue', alpha=0.1)
                
                in_position = True
            elif positions['position'].iloc[i] == 0 and in_position:
                # Exit point
                exit_price = data['Close'].iloc[i]
                ax.plot(positions.index[i], exit_price,
                    marker='o', markersize=8, color='purple', zorder=10)
                
                # Annotate trade outcome
                pnl = (exit_price - entry_price) / entry_price
                color = 'green' if pnl > 0 else 'red'
                ax.annotate(f'{pnl:.1%}', (positions.index[i], exit_price),
                        textcoords="offset points", xytext=(0,10),
                        ha='center', color=color, fontsize=10)
                
                in_position = False

    def _plot_market_structure(self, ax, data):
        """Plot market structure shifts."""
        structure_changes = data[data['market_structure'].abs() > 0]
        for idx, row in structure_changes.iterrows():
            if row['market_structure'] > 0:
                ax.axvline(idx, color='green', alpha=0.5, linestyle='--')
            else:
                ax.axvline(idx, color='red', alpha=0.5, linestyle='--')
        ax.set_title('Market Structure Shifts', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)

    def _plot_risk_parameters(self, ax, positions):
        """Plot position sizing and risk parameters."""
        ax.plot(positions['position'], label='Position Size', color='blue')
        ax.set_title('Risk Management Parameters', fontsize=14)
        ax.set_ylabel('Position Size', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    def _get_universe_tickers(self, universe='sp500'):
        """Get list of tickers for analysis."""
        if universe == 'sp500':
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            return table[0]['Symbol'].tolist()
        elif universe == 'nasdaq100':
            return pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]['Ticker'].tolist()
        else:
            raise ValueError("Unsupported universe")

    def generate_market_report(self, results):
        """Generate summary market report."""
        df = pd.DataFrame(results)
        df = df[df['current_signal'] == 'Long'].sort_values('return_1m', ascending=False)
        
        report = f"""
        SMC Market Analysis Report - {pd.Timestamp.today().date()}
        ------------------------------------------------
        Total Opportunities Found: {len(df)}
        Top 5 Long Candidates:
        {df.head(5).to_string(index=False)}
        
        Market Statistics:
        Average 1M Return: {df.return_1m.mean():.2%}
        Average Volatility: {df.volatility.mean():.2%}
        """
        
        with open('market_report.txt', 'w') as f:
            f.write(report)
            
        print(report)
