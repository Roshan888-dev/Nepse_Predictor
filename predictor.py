#!/usr/bin/env python3
"""
SMC Trading Model Module
------------------------
This module contains the feature engineering, model definition (an attention-based
transformer with LSTM and convolution layers), and predictor classes that implement
a universal trading system. The model training now includes more hyperparameters,
callbacks to help handle overfitting/underfitting, and an enhanced signal generation
logic that combines multiple SMC methodologies.
"""

from datetime import datetime
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from yfinance import Ticker, download

# =============================================================================
# SMC Feature Engineering
# =============================================================================
class SMCFeatureEngineer:
    def __init__(self, window=20):
        self.window = window
        self.scaler = StandardScaler()

    def calculate_features(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        # Basic Price Features
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.window).std()

        # SMC Features
        df['market_structure'] = self._calculate_market_structure(df)
        df['order_block_strength'] = self._calculate_order_blocks(df)
        df['liquidity_gradient'] = self._calculate_liquidity_zones(df)
        df['imbalance_ratio'] = self._calculate_imbalance_ratio(df)
        df['wyckoff_phase'] = self._calculate_wyckoff_phases(df)

        # Select and normalize features; use forward fill to avoid missing values.
        features = df[['market_structure', 'order_block_strength', 
                       'liquidity_gradient', 'imbalance_ratio', 
                       'wyckoff_phase', 'volatility']].ffill().dropna()
        return self.scaler.fit_transform(features)

    def _calculate_market_structure(self, df: pd.DataFrame) -> pd.Series:
        structure_series = pd.Series(0, index=df.index)
        highs = df['High']
        lows = df['Low']
        for i in range(2, len(df)-2):
            current_high = highs.iloc[i]
            prev1_high = highs.iloc[i-1]
            prev2_high = highs.iloc[i-2]
            current_low = lows.iloc[i]
            prev1_low = lows.iloc[i-1]
            prev2_low = lows.iloc[i-2]
            # Break of Structure (BOS)
            if current_high > prev1_high and current_high > prev2_high:
                structure_series.iloc[i] = 1  # Bullish BOS
            elif current_low < prev1_low and current_low < prev2_low:
                structure_series.iloc[i] = -1  # Bearish BOS
            # Change of Character (CHOCH)
            if (current_high < prev1_high and prev1_high > prev2_high and
                current_low > prev1_low and prev1_low < prev2_low):
                structure_series.iloc[i] = -2  # Bearish CHOCH
            elif (current_low > prev1_low and prev1_low < prev2_low and
                  current_high < prev1_high and prev1_high > prev2_high):
                structure_series.iloc[i] = 2  # Bullish CHOCH
        return structure_series

    def _calculate_order_blocks(self, df: pd.DataFrame) -> pd.Series:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        ob_strength = pd.Series(0.0, index=df.index)
        bearish_mask = (
            (df['Close'].shift(2) > df['Open'].shift(2)) &
            (df['Close'].shift(1) > df['Open'].shift(1)) &
            (df['Close'] < df['Open']) &
            (df['Volume'] > df['Volume'].shift(1) * 1.5)
        )
        bullish_mask = (
            (df['Close'].shift(2) < df['Open'].shift(2)) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Close'] > df['Open']) &
            (df['Volume'] > df['Volume'].shift(1) * 1.5)
        )
        ob_strength = ob_strength.mask(bullish_mask, 1.0)
        ob_strength = ob_strength.mask(bearish_mask, -1.0)
        return ob_strength

    def _calculate_liquidity_zones(self, df: pd.DataFrame) -> pd.Series:
        gradient = pd.Series(0.0, index=df.index)
        swing_highs = df['High'][(df['High'] > df['High'].shift(1)) & 
                                  (df['High'] > df['High'].shift(-1))]
        swing_lows = df['Low'][(df['Low'] < df['Low'].shift(1)) & 
                                (df['Low'] < df['Low'].shift(-1))]
        for idx in swing_highs.index:
            gradient.loc[idx] = 1.0
        for idx in swing_lows.index:
            gradient.loc[idx] = -1.0
        return gradient.rolling(5).mean().fillna(0)

    def _calculate_imbalance_ratio(self, df: pd.DataFrame) -> pd.Series:
        imbalance = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)-1):
            current_low = df['Low'].iloc[i]
            current_high = df['High'].iloc[i]
            next_high = df['High'].iloc[i+1]
            next_low = df['Low'].iloc[i+1]
            if current_high < next_low:
                imbalance.iloc[i] = 1.0
            elif current_low > next_high:
                imbalance.iloc[i] = -1.0
        return imbalance

    def _calculate_wyckoff_phases(self, df: pd.DataFrame) -> pd.Series:
        wyckoff = pd.Series(0, index=df.index)
        vwap = df['VWAP'] if 'VWAP' in df.columns else df['Close']
        for i in range(4, len(df)-4):
            # Accumulation Phase
            if (df['Volume'].iloc[i] > df['Volume'].iloc[i-1] and
                df['Close'].iloc[i] > vwap.iloc[i] and
                df['Low'].iloc[i] > df['Low'].iloc[i-2]):
                wyckoff.iloc[i] = 1
            # Distribution Phase
            elif (df['Volume'].iloc[i] > df['Volume'].iloc[i-1] and
                  df['Close'].iloc[i] < vwap.iloc[i] and
                  df['High'].iloc[i] < df['High'].iloc[i-2]):
                wyckoff.iloc[i] = -1
        return wyckoff

# =============================================================================
# SMC Neural Network Architecture (Attention Transformer)
# =============================================================================
class SMCTransformer(tf.keras.Model):
    def __init__(self, num_features, num_classes=3,
                 lstm_units=128, conv_filters=64, conv_kernel_size=3,
                 num_heads=4, key_dim=64, dropout_rate=0.3, dense_units=64,
                 attention_dropout=0.2, lstm_dropout=0.2):
        super().__init__()
        self.conv1d = tf.keras.layers.Conv1D(filters=conv_filters, 
                                               kernel_size=conv_kernel_size,
                                               activation='relu')
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, return_sequences=True, dropout=lstm_dropout))
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                            key_dim=key_dim,
                                                            dropout=attention_dropout)
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(dense_units, activation='gelu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs, training=False):
        x = self.conv1d(inputs)
        x = self.lstm(x, training=training)
        # Self-attention: query, key and value are the same in this case.
        x = self.attention(x, x, training=training)
        x = self.global_pool(x)
        return self.classifier(x)

# =============================================================================
# Universal SMC Predictor and Trainer
# =============================================================================
class UniversalSMCPredictor:
    def __init__(self,
                 risk_per_trade=0.01,
                 stop_loss_pct=0.05,
                 take_profit_pct=0.10,
                 transformer_params: dict = None,
                 training_params: dict = None,
                 signal_thresholds: dict = None):
        self.feature_engineer = SMCFeatureEngineer()
        # Transformer hyperparameters (with defaults)
        default_transformer_params = {
            'lstm_units': 128,
            'conv_filters': 64,
            'conv_kernel_size': 3,
            'num_heads': 4,
            'key_dim': 64,
            'dropout_rate': 0.3,
            'dense_units': 64,
            'attention_dropout': 0.2,
            'lstm_dropout': 0.2
        }
        if transformer_params:
            default_transformer_params.update(transformer_params)
        self.transformer_params = default_transformer_params

        # Build the model using the transformer parameters.
        self.model = self._build_model(num_features=6, **self.transformer_params)
        self.class_labels = ['Long', 'Neutral', 'Exit']
        self.risk_params = {
            'risk_per_trade': risk_per_trade,
            'stop_loss': stop_loss_pct,
            'take_profit': take_profit_pct
        }
        # Signal thresholds for generating buy/sell signals (tweak as needed)
        default_signal_thresholds = {
            'ob_long': 0.8,
            'ms_long': [1, 2],
            'liq_long': -0.7,
            'imb_long': 0.7,
            'wyk_long': 1,
            'ob_exit': -0.8,
            'ms_exit': [-1, -2],
            'liq_exit': 0.7,
            'imb_exit': -0.7,
            'wyk_exit': -1
        }
        self.signal_thresholds = signal_thresholds or default_signal_thresholds

        # Training parameters with defaults
        default_training_params = {
            'epochs': 50,
            'batch_size': 64,
            'validation_split': 0.2,
            'early_stopping_patience': 5,
            'optimizer': 'adamax'
        }
        self.training_params = training_params or default_training_params

    def _build_model(self, num_features, **kwargs):
        return SMCTransformer(num_features=num_features, **kwargs)

    def save_model(self, path: str = r'model.keras'):
        """Export trained model to specified path"""
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str = 'smc_model'):
        """Load pre-trained model"""
        self.model = tf.keras.models.load_model(path, custom_objects={'SMCTransformer': SMCTransformer})
        print("Model loaded successfully")

    def _generate_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Generate target labels based on future price movements (3-day horizon)"""
        future_returns = data['Close'].pct_change(3).shift(-3)
        labels = []
        for ret in future_returns:
            if ret > 0.02:
                labels.append(0)  # Buy signal
            elif ret < -0.02:
                labels.append(1)  # Sell signal
            else:
                labels.append(2)  # Hold / Neutral
        # Ensure alignment (drop lookahead periods)
        return np.array(labels)[:len(data) - 3]

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals by combining multiple SMC features."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        df = data.copy()

        # Calculate individual SMC features (if not already present)
        df['ob_strength'] = self.feature_engineer._calculate_order_blocks(df)
        df['market_structure'] = self.feature_engineer._calculate_market_structure(df)
        df['liquidity_gradient'] = self.feature_engineer._calculate_liquidity_zones(df)
        df['imbalance_ratio'] = self.feature_engineer._calculate_imbalance_ratio(df)
        df['wyckoff_phase'] = self.feature_engineer._calculate_wyckoff_phases(df)

        signals = []
        thresh = self.signal_thresholds
        # Loop through each row and apply combined SMC signal logic.
        for idx, row in df.iterrows():
            signal = 'Neutral'
            if any([
                row['ob_strength'] > thresh['ob_long'],
                row['market_structure'] in thresh['ms_long'],
                row['liquidity_gradient'] < thresh['liq_long'],
                row['imbalance_ratio'] > thresh['imb_long'],
                row['wyckoff_phase'] == thresh['wyk_long']
            ]):
                signal = 'Long'
            elif any([
                row['ob_strength'] < thresh['ob_exit'],
                row['market_structure'] in thresh['ms_exit'],
                row['liquidity_gradient'] > thresh['liq_exit'],
                row['imbalance_ratio'] < thresh['imb_exit'],
                row['wyckoff_phase'] == thresh['wyk_exit']
            ]):
                signal = 'Exit'
            signals.append(signal)
        return pd.Series(signals, index=df.index, name='signals')

    def calculate_positions(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """Calculate position sizes and risk parameters using proper indexing."""
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
                positions.at[current_idx, 'stop_loss'] = entry_price * (1 - self.risk_params['stop_loss'])
                positions.at[current_idx, 'take_profit'] = entry_price * (1 + self.risk_params['take_profit'])
                positions.at[current_idx, 'position'] = self.risk_params['risk_per_trade']
                in_position = True
            elif in_position:
                prev_idx = positions.index[i-1]
                # Exit if signal is 'Exit' or price reaches stop loss/take profit
                if (positions['signal'].iat[i] == 'Exit' or 
                    data['Low'].iat[i] < positions.at[prev_idx, 'stop_loss'] or
                    data['High'].iat[i] > positions.at[prev_idx, 'take_profit']):
                    positions.at[current_idx, 'position'] = 0.0
                    in_position = False
                else:
                    positions.at[current_idx, 'position'] = positions.at[prev_idx, 'position']
                    positions.at[current_idx, 'entry_price'] = positions.at[prev_idx, 'entry_price']
                    # Slight adjustment to stop loss and take profit to simulate trailing stops.
                    positions.at[current_idx, 'stop_loss'] = positions.at[prev_idx, 'stop_loss'] * 1.01
                    positions.at[current_idx, 'take_profit'] = positions.at[prev_idx, 'take_profit'] * 1.005
        return positions

    def plot_analysis(self, ticker: str, data: pd.DataFrame, signals: pd.Series, positions: pd.DataFrame):
        """Generate an analysis chart with signals, stop loss, and take profit annotations."""
        fig, axs = plt.subplots(2, 1, figsize=(20, 10), constrained_layout=True)
        ax1, ax2 = axs

        # Price chart with signals
        ax1.plot(data.index, data['Close'], '-', label='Price', color='blue', linewidth=1.5)
        buy_signals = signals == 'Long'
        sell_signals = signals == 'Exit'
        if any(buy_signals):
            ax1.plot(data.loc[buy_signals].index, data.loc[buy_signals, 'Close'],
                     '^', color='green', ms=10, label='Buy Signal')
        if any(sell_signals):
            ax1.plot(data.loc[sell_signals].index, data.loc[sell_signals, 'Close'],
                     'v', color='red', ms=10, label='Sell Signal')

        # Plot stop loss and take profit levels
        valid_stops = positions['stop_loss'].notna()
        valid_profits = positions['take_profit'].notna()
        if any(valid_stops):
            ax1.plot(positions.loc[valid_stops].index, positions.loc[valid_stops, 'stop_loss'],
                     '--', color='maroon', label='Stop Loss')
        if any(valid_profits):
            ax1.plot(positions.loc[valid_profits].index, positions.loc[valid_profits, 'take_profit'],
                     '--', color='darkblue', label='Take Profit')

        ax1.set_title(f'{ticker} Analysis - SMC Model', fontsize=16, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        # Trend Analysis (Moving Averages)
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
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        os.makedirs('analysis_charts', exist_ok=True)
        plt.savefig(f'analysis_charts/{ticker}_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _get_universe_tickers(self, universe: str = 'sp500') -> list:
        """Get list of tickers from a given universe."""
        if universe == 'sp500':
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            return table[0]['Symbol'].tolist()
        elif universe == 'nasdaq100':
            return pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]['Ticker'].tolist()
        else:
            raise ValueError("Unsupported universe")

    def generate_market_report(self, results: list):
        """Generate a summary market report."""
        df = pd.DataFrame(results)
        df = df[df['current_signal'] == 'Long'].sort_values('return_1m', ascending=False)
        report = f"""
        SMC Market Analysis Report - {datetime.today().date()}
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

    def _create_sequences(self, features: np.ndarray, labels: np.ndarray, window: int = 30):
        sequences = []
        seq_labels = []
        for i in range(len(features) - window):
            sequences.append(features[i:i + window])
            seq_labels.append(labels[i + window])
        return np.array(sequences), np.array(seq_labels)

    def train_universal(self, stock_list: list, start_date: str, end_date: str):
        """Train the SMC model using historical data for a list of stocks."""
        all_features = []
        all_labels = []
        for ticker in stock_list:
            raw_data = Ticker(ticker).history(start=start_date, end=end_date)
            if raw_data.empty or len(raw_data) < 100:
                continue
            features = self.feature_engineer.calculate_features(raw_data)
            labels = self._generate_labels(raw_data)
            # Align lengths
            min_length = min(len(features), len(labels))
            features = features[:min_length]
            labels = labels[:min_length]
            seq_features, seq_labels = self._create_sequences(features, labels)
            all_features.append(seq_features)
            all_labels.append(seq_labels)
        if not all_features:
            print("No valid data for training!")
            return

        X = np.concatenate(all_features)
        y = np.concatenate(all_labels)
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.training_params.get('validation_split', 0.2))
        # Compile the model with chosen optimizer and loss.
        self.model.compile(optimizer=self.training_params.get('optimizer', 'adamax'),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        # Callbacks to help handle overfitting
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=self.training_params.get('early_stopping_patience', 5),
                                             restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
        ]
        history = self.model.fit(X_train, y_train,
                                 epochs=self.training_params.get('epochs', 50),
                                 batch_size=self.training_params.get('batch_size', 64),
                                 validation_data=(X_test, y_test),
                                 callbacks=callbacks)
        # Evaluate model performance
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Model Evaluation -- Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return history

    def predict_stock(self, ticker: str) -> str:
        """Generate a trading signal prediction for a given stock ticker."""
        data = Ticker(ticker).history(period="max")
        features = self.feature_engineer.calculate_features(data)
        # Create a sequence from the last 30 observations
        latest_sequence = features[-30:].reshape(1, 30, -1)
        prediction = self.model.predict(latest_sequence)
        return self.class_labels[np.argmax(prediction)]

    def analyze_market(self, universe: str = 'sp500', year_back: int = 1, batch_size: int = 50) -> list:
        """Analyze an entire market universe and generate analysis charts."""
        tickers = self._get_universe_tickers(universe)
        analysis_results = []
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            for ticker in batch_tickers:
                try:
                    data = download(ticker, period=f'{year_back}y')
                    if len(data) < 100:
                        continue
                    df = data.copy()
                    features = self.feature_engineer.calculate_features(df)
                    signals = self.generate_signals(df)
                    positions = self.calculate_positions(df, signals)
                    self.plot_analysis(ticker, df, signals, positions)
                    analysis_results.append({
                        'ticker': ticker,
                        'current_signal': signals.iloc[-1],
                        'return_1m': df['Close'].pct_change(21).iloc[-1],
                        'volatility': df['Close'].pct_change().std() * np.sqrt(252)
                    })
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
            plt.close('all')
        self.generate_market_report(analysis_results)
        return analysis_results

# =============================================================================
# Main Execution Block (Usage Example)
# =============================================================================
if __name__ == "__main__":
    # Initialize the predictor with optional hyperparameter updates
    transformer_hyperparams = {
        'lstm_units': 128,
        'conv_filters': 64,
        'conv_kernel_size': 3,
        'num_heads': 4,
        'key_dim': 64,
        'dropout_rate': 0.4,         # Increased dropout to help regularization
        'dense_units': 64,
        'attention_dropout': 0.3,
        'lstm_dropout': 0.3
    }
    training_hyperparams = {
        'epochs': 60,
        'batch_size': 64,
        'validation_split': 0.2,
        'early_stopping_patience': 6,
        'optimizer': 'adamax'
    }
    smc = UniversalSMCPredictor(transformer_params=transformer_hyperparams,
                                training_params=training_hyperparams)

    # Try to load a pre-trained model. If not available, train a new one.
    try:
        smc.load_model('model.keras')
    except Exception as e:
        print(f"Loading model failed: {e}\nTraining new model...")
        # Example: training on a small list of ETFs. Extend as needed.
        smc.train_universal(['SPY', 'QQQ'], '2010-01-01', '2023-01-01')
        smc.save_model('model.keras')

    # Perform a full market analysis
    market_results = smc.analyze_market(universe='sp500')

    # Generate signals and positions for the top candidate
    if market_results:
        top_pick = market_results[0]['ticker']
        data = download(top_pick, period='1y')
        signals = smc.generate_signals(data)
        positions = smc.calculate_positions(data, signals)
        print(f"\nRecommended Position for {top_pick}:")
        print(positions.tail(10))
    else:
        print("No market opportunities found.")
