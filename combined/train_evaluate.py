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
        """Generate analysis charts."""
        fig, axs = plt.subplots(2, 1, figsize=(20, 10), constrained_layout=True)
        
        # Main Price Chart
        ax1 = axs[0]
        ax1.plot(data.index, data['Close'], '-', label='Price', color='blue', linewidth=1.5)
        
        buy_signals = signals == 'Long'
        sell_signals = signals == 'Exit'
        
        if any(buy_signals):
            ax1.plot(data.loc[buy_signals].index, 
                    data.loc[buy_signals, 'Close'],
                    '^', color='green', ms=10, label='Buy Signal')
        
        if any(sell_signals):
            ax1.plot(data.loc[sell_signals].index,
                    data.loc[sell_signals, 'Close'],
                    'v', color='red', ms=10, label='Sell Signal')
        
        valid_stops = positions['stop_loss'].notna()
        valid_profits = positions['take_profit'].notna()
        
        if any(valid_stops):
            ax1.plot(positions.loc[valid_stops].index,
                    positions.loc[valid_stops, 'stop_loss'],
                    '--', color='maroon', label='Stop Loss')
        
        if any(valid_profits):
            ax1.plot(positions.loc[valid_profits].index,
                    positions.loc[valid_profits, 'take_profit'],
                    '--', color='darkblue', label='Take Profit')
        
        ax1.set_title(f'{ticker} Analysis - SMC Model', fontsize=16, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Trend Analysis
        ax2 = axs[1]
        ma50 = data['Close'].rolling(50).mean()
        ma200 = data['Close'].rolling(200).mean()
        min50 = data['Close'].rolling(50).min()
        max50 = data['Close'].rolling(50).max()
        
        ax2.plot(ma50.index, ma50, '-', label='50 DMA', color='orange', linewidth=1.5)
        ax2.plot(ma200.index, ma200, '-', label='200 DMA', color='purple', linewidth=1.5)
        ax2.fill_between(data.index, min50, max50, color='gray', alpha=0.2, label='50 Period Range')
        
        ax2.set_title('Trend Analysis', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        os.makedirs('analysis_charts', exist_ok=True)
        plt.savefig(f'analysis_charts/{ticker}_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

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
