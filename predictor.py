from datetime import datetime
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from yfinance import Ticker, download
import matplotlib.dates as mdates
import matplotlib

from smc_trading.config import PATHS
matplotlib.use('Agg') 
# Enhanced SMC Feature Extractor
class SMCFeatureEngineer:
    def __init__(self, window=20):
        self.window = window
        self.scaler = StandardScaler()
        
    def calculate_features(self, df):
        # Remove df.copy() to modify the original DataFrame
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.window).std()
        df['market_structure'] = self._calculate_market_structure(df)
        df['order_block_strength'] = self._calculate_order_blocks(df)
        df['liquidity_gradient'] = self._calculate_liquidity_zones(df)
        df['imbalance_ratio'] = self._calculate_imbalance_ratio(df)
        df['wyckoff_phase'] = self._calculate_wyckoff_phases(df)
        df['atr'] = self._calculate_atr(df)

        # Validate ATR calculation
        if df['atr'].isnull().all():
            raise ValueError("ATR calculation failed")

        # Include 'atr' in the features to be scaled
        features = df[['market_structure', 'order_block_strength', 
                    'liquidity_gradient', 'imbalance_ratio', 
                    'wyckoff_phase', 'volatility', 'atr']].ffill().dropna()

        # Scale the features
        return self.scaler.fit_transform(features)
    
    def _calculate_atr(self, df, period=14):
        """Robust ATR calculation with error handling"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            tr = np.maximum(
                high - low,
                np.maximum(
                    abs(high - close.shift()),
                    abs(low - close.shift())
                )
            )
            atr = tr.rolling(period, min_periods=1).mean().bfill()
            print(f"ATR calculated successfully for {len(atr)} rows.")  # Debug statement
            return atr
        except KeyError as e:
            print(f"Missing OHLC columns in DataFrame: {e}")
            return pd.Series(np.nan, index=df.index)
    
    def _calculate_market_structure(self, df):
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
            if current_high > prev2_high and current_high > prev1_high:
                structure_series.iloc[i] = 1  # Bullish BOS code
            elif current_low < prev2_low and current_low < prev1_low:
                structure_series.iloc[i] = -1  # Bearish BOS code
            
            # Change of Character (CHOCH)
            if (current_high < prev1_high and prev1_high > prev2_high and
                current_low > prev1_low and prev1_low < prev2_low):
                structure_series.iloc[i] = -2  # Bearish CHOCH code
            elif (current_low > prev1_low and prev1_low < prev2_low and
                current_high < prev1_high and prev1_high > prev2_high):
                structure_series.iloc[i] = 2  # Bullish CHOCH code
        
        return structure_series
    
    def _calculate_order_blocks(self, df):
        """Improved order block calculation with DataFrame validation"""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        ob_strength = pd.Series(0.0, index=df.index)
        
        # Vectorized operations for better performance
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

    def _calculate_liquidity_zones(self, df):
        gradient = pd.Series(0.0, index=df.index)
        swing_highs = df['High'][(df['High'] > df['High'].shift(1)) & 
                                (df['High'] > df['High'].shift(-1))]
        swing_lows = df['Low'][(df['Low'] < df['Low'].shift(1)) & 
                              (df['Low'] < df['Low'].shift(-1))]
        
        # Calculate gradient based on swing points
        for idx in swing_highs.index:
            gradient.loc[idx] = 1.0
        for idx in swing_lows.index:
            gradient.loc[idx] = -1.0
            
        return gradient.rolling(5).mean().fillna(0)

    def _calculate_imbalance_ratio(self, df):
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

    def _calculate_wyckoff_phases(self, df):
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

# SMC Neural Network Architecture
class SMCTransformer(tf.keras.Model):
    def __init__(self, num_features, num_classes=3):
        super().__init__()
        
        # Enhanced Multi-Head Attention Mechanism
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=64,
            dropout=0.1
        )
        
        # Bidirectional LSTM for capturing temporal dependencies
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)
        )
        
        # 1D Convolutional Layer for feature extraction
        self.conv1d = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')
        
        # Layer Normalization for stabilizing training
        self.layer_norm = tf.keras.layers.LayerNormalization()
        
        # Feedforward Network for classification
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='gelu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Positional Encoding for time series data
        self.positional_encoding = self._create_positional_encoding(
            max_len=100, 
            d_model=num_features
        )
        
    def _create_positional_encoding(self, max_len, d_model):
        """Generate positional encoding with proper shape handling."""
        position = np.arange(max_len)[:, np.newaxis]
        
        # Ensure even number of dimensions for sin/cos pairs
        d_even = (d_model // 2) * 2
        div_term = np.exp(np.arange(0, d_even, 2) * -(np.log(10000.0) / d_even))
        
        # Initialize pe with zeros for the full d_model dimension
        pe = np.zeros((max_len, d_model))
        
        # Calculate sin and cos components for even indices
        pe[:, 0:d_even:2] = np.sin(position * div_term)
        pe[:, 1:d_even:2] = np.cos(position * div_term)
        
        # If d_model is odd, handle the last dimension separately
        if d_model % 2 == 1:
            last_dim = np.sin(position * div_term[-1])
            pe[:, -1] = last_dim[:, 0]
        
        return tf.constant(pe, dtype=tf.float32)
    
    def call(self, inputs):
        # Add positional encoding to inputs
        seq_len = tf.shape(inputs)[1]
        inputs += self.positional_encoding[:seq_len, :]
        
        # Apply 1D Convolution
        x = self.conv1d(inputs)
        
        # Apply Bidirectional LSTM
        x = self.lstm(x)
        
        # Apply Multi-Head Attention with residual connection
        attn_output = self.attention(x, x)
        x = self.layer_norm(x + attn_output)
        
        # Global Average Pooling to reduce sequence dimension
        x = tf.reduce_mean(x, axis=1)
        
        # Final classification
        return self.classifier(x)

# Universal SMC Predictor
class UniversalSMCPredictor:
    def __init__(self, risk_per_trade=0.01, stop_loss_pct=0.05, take_profit_pct=0.10):
        self.feature_engineer = SMCFeatureEngineer()
        self.model = self._build_model()
        self.class_labels = ['Long', 'Neutral', 'Exit']
        self.risk_params = {
            'risk_per_trade': risk_per_trade,
            'stop_loss': stop_loss_pct,
            'take_profit': take_profit_pct
        }
        
    def save_model(self, path=r'C:\Roshan\nepse predictor\model.keras'):
        """Export trained model to specified path"""
        self.model.save(path)
        print(f"Model saved to {path}")


    def load_model(self, path='smc_model'):
        """Load pre-trained model"""
        self.model = tf.keras.models.load_model(path)
        print("Model loaded successfully")

    def _build_model(self):
        return SMCTransformer(num_features=7)

    def _generate_labels(self, data):
        """Generate target labels based on future price movements (3-day horizon)"""
        labels = []
        
        # Calculate future returns
        future_returns = data['Close'].pct_change(3).shift(-3)
        
        # Create labels based on return thresholds
        for ret in future_returns:
            if ret > 0.02:  # 2% gain threshold
                labels.append(0)  # Buy
            elif ret < -0.02:  # 2% loss threshold
                labels.append(1)  # Sell
            else:
                labels.append(2)  # Hold
        
        # Convert to numpy array and align with features
        labels = np.array(labels)[:len(data)-3]  # Account for lookahead
        return labels
    
    def generate_signals(self, data):
        """Generate trading signals with proper DataFrame validation"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        
        signals = []
        df = data.copy()
        
        # Calculate features using DataFrame columns
        df['ob_strength'] = self.feature_engineer._calculate_order_blocks(df)
        df['market_structure'] = self.feature_engineer._calculate_market_structure(df)
        df['liquidity_gradient'] = self.feature_engineer._calculate_liquidity_zones(df)
        df['imbalance_ratio'] = self.feature_engineer._calculate_imbalance_ratio(df)
        df['wyckoff_phase'] = self.feature_engineer._calculate_wyckoff_phases(df)

        # Use iterrows() to iterate over DataFrame rows
        for idx, row in df.iterrows():
            signal = 'Neutral'  # Default state
            
            # Access values directly from the row
            ob = row['ob_strength']
            ms = row['market_structure']
            liq = row['liquidity_gradient']
            imb = row['imbalance_ratio']
            wyk = row['wyckoff_phase']

            # Signal logic with explicit conditions
            if any([
                ob > 0.8,
                ms in [1, 2],
                liq < -0.7,
                imb > 0.7,
                wyk == 1
            ]):
                signal = 'Long'
            elif any([
                ob < -0.8,
                ms in [-1, -2],
                liq > 0.7,
                imb < -0.7,
                wyk == -1
            ]):
                signal = 'Exit'
                
            signals.append(signal)
        
        return pd.Series(signals, index=df.index, name='signals')

    def analyze_market(self, universe='sp500', year_back=1, batch_size=50):
        tickers = self._get_universe_tickers(universe)
        analysis_results = []
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            for ticker in batch_tickers:
                try:
                    data = download(ticker, period=f'{year_back}y')
                    if len(data) < 100:
                        continue
                    
                    # Calculate features
                    features = self.feature_engineer.calculate_features(data)
                    signals = self.generate_signals(data)
                    
                    # Ensure 'atr' is in the DataFrame
                    if 'atr' not in data.columns:
                        raise ValueError("ATR column missing in DataFrame")
                    
                    positions = self.calculate_positions(data, signals)
                    
                    # Generate analysis chart
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

    def calculate_positions(self, data, signals):
        """Calculate position sizes with volatility-adjusted stops"""
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
                atr = data['atr'].iat[i]
                
                positions.at[current_idx, 'entry_price'] = entry_price
                positions.at[current_idx, 'stop_loss'] = entry_price - 1.5 * atr
                positions.at[current_idx, 'take_profit'] = entry_price + 2.5 * atr
                positions.at[current_idx, 'position'] = min(
                    self.risk_params['risk_per_trade'] / (1.5 * atr / entry_price),
                    0.1  # Max position size 10%
                )
                in_position = True
                
            elif in_position:
                prev_idx = positions.index[i-1]
                current_high = data['High'].iat[i]
                current_low = data['Low'].iat[i]
                
                # Trailing stop logic
                new_stop = max(positions.at[prev_idx, 'stop_loss'], 
                              current_high - 0.5 * data['atr'].iat[i])
                
                if positions['signal'].iat[i] == 'Exit' or \
                current_low < positions.at[prev_idx, 'stop_loss'] or \
                current_high > positions.at[prev_idx, 'take_profit']:
                    positions.at[current_idx, 'position'] = 0.0
                    in_position = False
                else:
                    positions.at[current_idx, 'position'] = positions.at[prev_idx, 'position']
                    positions.at[current_idx, 'entry_price'] = positions.at[prev_idx, 'entry_price']
                    positions.at[current_idx, 'stop_loss'] = new_stop
                    positions.at[current_idx, 'take_profit'] = positions.at[prev_idx, 'take_profit'] * 1.005
                    
        return positions

    def plot_analysis(self, ticker, data, signals, positions):
        """Generate clean, responsive charts with detailed position info"""
        plt.style.use('seaborn-dark')
        fig, axs = plt.subplots(3, 1, figsize=(24, 18), 
                              gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Price Chart with Position Markers
        ax1 = axs[0]
        ax1.plot(data.index, data['Close'], label='Price', color='royalblue', lw=1.5)
        
        # Plot entry and exit points
        entries = positions[positions['position'] > 0]
        exits = positions[positions['position'] == 0]
        
        ax1.scatter(entries.index, entries['entry_price'],
                   marker='^', s=100, color='limegreen', edgecolor='black',
                   label='Entry', zorder=5)
        
        ax1.scatter(exits.index, data.loc[exits.index, 'Close'],
                   marker='v', s=100, color='crimson', edgecolor='black',
                   label='Exit', zorder=5)
        
        # Plot dynamic stops and targets
        ax1.plot(positions.index, positions['stop_loss'],
                color='darkred', ls='--', lw=1.2, label='Stop Loss')
        ax1.plot(positions.index, positions['take_profit'],
                color='darkgreen', ls='--', lw=1.2, label='Take Profit')
        
        ax1.set_title(f'{ticker} Trading Analysis', fontsize=16, pad=20)
        ax1.legend(loc='best', fontsize=10)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.tick_params(axis='x', rotation=45)
        
        # Position Size Chart
        ax2 = axs[1]
        ax2.fill_between(positions.index, positions['position'], 
                        color='teal', alpha=0.3, label='Position Size')
        ax2.set_ylabel('Position Size', fontsize=10)
        ax2.grid(True, alpha=0.4)
        
        # Volatility Chart
        ax3 = axs[2]
        ax3.plot(data.index, data['atr'], 
                color='purple', lw=1.2, label='ATR (Volatility)')
        ax3.set_ylabel('ATR', fontsize=10)
        ax3.grid(True, alpha=0.4)
        
        # Save high-res responsive chart
        plt.savefig(f'analysis_charts/{ticker}_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


    def _get_universe_tickers(self, universe='sp500'):
        """Get list of tickers for analysis"""
        if universe == 'sp500':
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            return table[0]['Symbol'].tolist()
        elif universe == 'nasdaq100':
            return pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]['Ticker'].tolist()
        else:
            raise ValueError("Unsupported universe")

    def generate_market_report(self, results):
        """Generate validated market report with error handling"""
        if not results:
            print("No results to generate report")
            return

        try:
            df = pd.DataFrame(results)
            
            # Initialize default values
            report_content = f"""
            SMC Market Analysis Report - {datetime.today().date()}
            {'-'*50}
            Total Assets Analyzed: {len(df)}
            """
            
            # Check for signals and returns
            if 'current_signal' not in df.columns:
                report_content += "\nWarning: No valid signals generated"
            else:
                long_candidates = df[df['current_signal'] == 'Long']
                report_content += f"""
                Trading Opportunities Found: {len(long_candidates)}
                """
                
                if not long_candidates.empty:
                    report_content += """
                    Top 5 Candidates:
                    """ + long_candidates.sort_values('return_1m', ascending=False)\
                        .head(5).to_string(index=False)
            
            # Add market statistics if available
            if 'volatility' in df.columns:
                report_content += f"""
                Market Statistics:
                Average 1M Return: {df['return_1m'].mean():.2%}
                Average Volatility: {df['volatility'].mean():.2%}
                """
                
            # Save and display report
            print(report_content)
            os.makedirs(PATHS['reports'], exist_ok=True)
            with open(os.path.join(PATHS['reports'], 'market_report.txt'), 'w') as f:
                f.write(report_content)
                
        except Exception as e:
            print(f"Failed to generate report: {str(e)}")

    def train_universal(self, stock_list, start_date, end_date):
        all_features = []
        all_labels = []
        
        for ticker in stock_list:
            try:
                # Get data with proper date range handling
                data = download(ticker, start=start_date, end=end_date)
                if data.empty or len(data) < 100:
                    continue
                    
                # Calculate features with validation
                features = self.feature_engineer.calculate_features(data)
                labels = self._generate_labels(data)
                
                if len(features) < 30 or len(labels) < 30:
                    continue
                    
                # Create sequences with length validation
                seqs, lbls = self._create_sequences(features, labels)
                if len(seqs) > 0:
                    all_features.append(seqs)
                    all_labels.append(lbls)
                    
            except Exception as e:
                print(f"Skipped {ticker}: {str(e)}")
                continue
        
        # Validate training data
        if not all_features:
            raise ValueError("No valid training data accumulated")
        
        # Concatenate with safety checks
        X = np.concatenate(all_features) if all_features else np.array([])
        y = np.concatenate(all_labels) if all_labels else np.array([])
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Insufficient training data")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        self.model.compile(optimizer='adamax',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=50, batch_size=64,
                    validation_data=(X_test, y_test))

    def _create_sequences(self, features, labels, window=30):
        sequences = []
        seq_labels = []
        for i in range(len(features)-window):
            sequences.append(features[i:i+window])
            seq_labels.append(labels[i+window])
        return np.array(sequences), np.array(seq_labels)

    def predict_stock(self, ticker):
        data = Ticker(ticker).history(period="max")
        features = self.feature_engineer.calculate_features(data)
        latest_sequence = features[-30:].reshape(1, 30, -1)
        prediction = self.model.predict(latest_sequence)
        return self.class_labels[np.argmax(prediction)]

# Usage Example
if __name__ == "__main__":
    smc = UniversalSMCPredictor(risk_per_trade=0.02)
    
    try:
        smc.load_model()
    except:
        smc.train_universal(['SPY', 'QQQ'], '2010-01-01', '2023-01-01')
        smc.save_model()
    
    results = smc.analyze_market(universe='sp500')
    
    # Check if results are available
    if not results:
        print("No results to generate report or process further.")
    else:
        # Generate positions for top candidate
        top_pick = results[0]['ticker']
        data = download(top_pick, period='1y')
        features = smc.feature_engineer.calculate_features(data)
        signals = smc.generate_signals(data)
        positions = smc.calculate_positions(data, signals)
        
        print(f"\nRecommended Position for {top_pick}:")
        print(positions.tail())