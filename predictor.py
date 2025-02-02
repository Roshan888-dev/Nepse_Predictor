from datetime import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from yfinance import Ticker, download
import matplotlib.dates as mdates
import matplotlib
from typing import Dict, List, Optional, Tuple

matplotlib.use('Agg')

# Configuration paths (ensure this exists in your project)
class PATHS:
    reports = 'analysis_reports'
# Enhanced SMC Feature Extractor
class SMCFeatureEngineer:
    def __init__(self, window=20):
        self.window = window
        self.scaler = StandardScaler()
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if df.isnull().values.any():
            df.ffill(inplace=True)

    def calculate_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        self.validate_dataframe(df)
        
        df = df.copy()
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.window, min_periods=1).std()
        df['atr'] = self._calculate_atr(df)  # Now part of the main DataFrame
        
        # Calculate other features
        df['market_structure'] = self._calculate_market_structure(df)
        df['order_block_strength'] = self._calculate_order_blocks(df)
        df['liquidity_gradient'] = self._calculate_liquidity_zones(df)
        df['imbalance_ratio'] = self._calculate_imbalance_ratio(df)
        df['wyckoff_phase'] = self._calculate_wyckoff_phases(df)
        
        # Clean and align data
        feature_columns = ['market_structure', 'order_block_strength', 
                        'liquidity_gradient', 'imbalance_ratio', 
                        'wyckoff_phase', 'volatility', 'atr']
        processed_df = df[feature_columns].ffill().dropna()
        aligned_data = df.loc[processed_df.index]  # Contains all original columns + features
        
        # Scale features
        features = self.scaler.fit_transform(processed_df)
        
        return features, aligned_data

    def _calculate_atr(self, df: pd.DataFrame, period=14) -> pd.Series:
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            # Calculate True Range (TR)
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            
            # Calculate ATR with fallback to simple volatility
            atr = tr.rolling(period, min_periods=1).mean().bfill()
            return atr.fillna(high - low)  # Fallback if ATR calculation fails
        except KeyError as e:
            print(f"Missing OHLC columns: {e}")
            return (df['High'] - df['Low']).fillna(0)  # Simple volatility
    
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
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=64, dropout=0.1)
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2))
        self.conv1d = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='gelu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        self.positional_encoding = self._create_positional_encoding(100, num_features)

    def _create_positional_encoding(self, max_len, d_model):
        position = np.arange(max_len)[:, np.newaxis]
        d_even = (d_model // 2) * 2
        div_term = np.exp(np.arange(0, d_even, 2) * -(np.log(10000.0) / d_even))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0:d_even:2] = np.sin(position * div_term)
        pe[:, 1:d_even:2] = np.cos(position * div_term)
        
        if d_model % 2 == 1:
            pe[:, -1] = np.sin(position * div_term[-1])[:, 0]
            
        return tf.constant(pe, dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        inputs += self.positional_encoding[:seq_len, :]
        x = self.conv1d(inputs)
        x = self.lstm(x)
        attn_output = self.attention(x, x)
        x = self.layer_norm(x + attn_output)
        x = tf.reduce_mean(x, axis=1)
        return self.classifier(x)


class PositionManager:
    def __init__(self, risk_params: Dict):
        self.risk_params = risk_params

    def calculate_position_size(self, entry_price: float, atr: float) -> float:
        risk_per_share = atr * self.risk_params['stop_loss_multiplier']
        return min(
            self.risk_params['risk_per_trade'] / risk_per_share,
            self.risk_params['max_position_size']
        )

    def calculate_stop_levels(self, entry_price: float, atr: float, direction: str) -> Tuple[float, float]:
        if direction == 'long':
            return (
                entry_price - atr * self.risk_params['stop_loss_multiplier'],
                entry_price + atr * self.risk_params['take_profit_multiplier']
            )
        return (
            entry_price + atr * self.risk_params['stop_loss_multiplier'],
            entry_price - atr * self.risk_params['take_profit_multiplier']
        )

# Universal SMC Predictor
class UniversalSMCPredictor:
    def __init__(self, risk_params: Optional[Dict] = None):
        self.feature_engineer = SMCFeatureEngineer()
        self.model = self._build_model()
        self.class_labels = ['Long', 'Neutral', 'Exit']
        self.position_manager = PositionManager(
            risk_params or {
                'risk_per_trade': 0.01,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 3.0,
                'max_position_size': 0.1
            }
        )

    def save_model(self, path=r'C:\Roshan\nepse predictor\trained_models\model.keras'):
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Create directory if missing
        self.model.save(path, save_format='keras')
        print(f"Model saved to {path}")

    def load_model(self, path=r'C:\Roshan\nepse predictor\trained_models\model.keras'):
        try:
            self.model = tf.keras.models.load_model(path)
            print("Model loaded successfully")
        except (OSError, ValueError) as e:
            print(f"Error loading model: {str(e)}")
            raise

    def _build_model(self):
        return SMCTransformer(num_features=7)

    def _generate_labels(self, data):
        """Generate target labels based on future price movements (3-day horizon)"""
        labels = []
        future_returns = data['Close'].pct_change(3).shift(-3)
        
        # Create labels based on return thresholds
        for ret in future_returns:
            if ret > 0.02:  # 2% gain threshold
                labels.append(0)  # Buy
            elif ret < -0.02:  # 2% loss threshold
                labels.append(1)  # Sell
            else:
                labels.append(2)  # Hold
        
        # Trim the last 3 entries which have NaN future_returns
        labels = np.array(labels)[:-3]
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
            for ticker in tickers:
                try:
                    raw_data = download(ticker, period=f'{year_back}y')
                    
                    # Basic data validation
                    if raw_data.empty or len(raw_data) < 100:
                        print(f"Skipped {ticker}: insufficient data")
                        continue
                        
                    # Feature engineering
                    features, aligned_data = self.feature_engineer.calculate_features(raw_data)
                    
                    # Post-feature validation
                    if 'atr' not in aligned_data.columns:
                        print(f"Skipped {ticker}: ATR missing after feature engineering")
                        continue
                    
                    # Generate signals and positions using aligned_data
                    signals = self.generate_signals(aligned_data)
                    positions = self.calculate_positions(aligned_data, signals)
                    
                    # Generate analysis chart
                    self.plot_analysis(ticker, aligned_data, signals, positions)
                    
                    analysis_results.append({
                        'ticker': ticker,
                        'current_signal': signals.iloc[-1],
                        'return_1m': aligned_data['Close'].pct_change(21).iloc[-1],
                        'volatility': aligned_data['Close'].pct_change().std() * np.sqrt(252)
                    })
                    
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
                    continue
            
            plt.close('all')
        
        self.generate_market_report(analysis_results)
        return analysis_results

    def calculate_positions(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        positions = pd.DataFrame(index=data.index)
        positions['signal'] = signals
        positions['position'] = 0.0
        positions['entry_price'] = np.nan
        positions['stop_loss'] = np.nan
        positions['take_profit'] = np.nan
        current_position = None

        for i in range(1, len(positions)):
            current_idx = positions.index[i]
            prev_idx = positions.index[i-1]

            if not current_position and positions['signal'].iloc[i] == 'Long':
                self._enter_position(data, positions, current_idx, i, 'long')
                current_position = 'long'

            elif current_position == 'long':
                self._update_position(data, positions, current_idx, prev_idx, i, 'long')

        return positions

    def _enter_position(self, data: pd.DataFrame, positions: pd.DataFrame, 
                       idx: pd.Timestamp, i: int, direction: str):
        entry_price = data['Close'].iloc[i]
        atr = data['atr'].iloc[i]
        position_size = self.position_manager.calculate_position_size(entry_price, atr)
        stop_loss, take_profit = self.position_manager.calculate_stop_levels(
            entry_price, atr, direction
        )

        positions.at[idx, 'entry_price'] = entry_price
        positions.at[idx, 'position'] = position_size
        positions.at[idx, 'stop_loss'] = stop_loss
        positions.at[idx, 'take_profit'] = take_profit

    def _update_position(self, data: pd.DataFrame, positions: pd.DataFrame,
                        current_idx: pd.Timestamp, prev_idx: pd.Timestamp,
                        i: int, direction: str):
        current_high = data['High'].iloc[i]
        current_low = data['Low'].iloc[i]
        atr = data['atr'].iloc[i]

        new_stop = max(positions.at[prev_idx, 'stop_loss'], 
                      current_high - 0.5 * atr)
        new_take = positions.at[prev_idx, 'take_profit'] * 1.005

        if current_low < new_stop or current_high > new_take:
            positions.at[current_idx, 'position'] = 0.0
            return

        positions.at[current_idx, 'position'] = positions.at[prev_idx, 'position']
        positions.at[current_idx, 'entry_price'] = positions.at[prev_idx, 'entry_price']
        positions.at[current_idx, 'stop_loss'] = new_stop
        positions.at[current_idx, 'take_profit'] = new_take

    def plot_analysis(self, ticker, data, signals, positions):
        """Generate clean, responsive charts with detailed position info"""
        plt.style.use('seaborn-v0_8-dark')# Updated style name
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
        os.makedirs('analysis_charts', exist_ok=True)  # Ensure directory exists
        plt.savefig(f'analysis_charts/{ticker}_analysis.png', 
                dpi=300, bbox_inches='tight')
        plt.close()


    def _get_universe_tickers(self, universe='sp500'):
        """Get list of tickers for analysis, skipping invalid or delisted tickers"""
        if universe == 'sp500':
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            tickers = table[0]['Symbol'].tolist()
        elif universe == 'nasdaq100':
            tickers = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]['Ticker'].tolist()
        else:
            raise ValueError("Unsupported universe")
        
        # Remove invalid tickers (e.g., BF.B)
        valid_tickers = [ticker for ticker in tickers if '.' not in ticker]
        return valid_tickers

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
                data = download(ticker, start=start_date, end=end_date)
                if data.empty or len(data) < 100:
                    print(f"Skipped {ticker}: Insufficient data")
                    continue
                    
                # Validate DataFrame columns
                self.feature_engineer.validate_dataframe(data)
                    
                # Get features and aligned data
                features, aligned_data = self.feature_engineer.calculate_features(data)
                labels = self._generate_labels(aligned_data)
                
                # Check for sufficient data after processing
                if len(features) < 30 or len(labels) < 30:
                    print(f"Skipped {ticker}: Insufficient data after processing")
                    continue
                    
                # Create sequences with correct alignment
                seqs, lbls = self._create_sequences(features, labels)
                if len(seqs) > 0:
                    all_features.append(seqs)
                    all_labels.append(lbls)
                    
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue
        
        if not all_features:
            raise ValueError("No valid training data accumulated")
        
        X = np.concatenate(all_features)
        y = np.concatenate(all_labels)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        self.model.compile(optimizer='adamax',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=50, batch_size=64,
                    validation_data=(X_test, y_test))

    def _create_sequences(self, features, labels, window=30):
        sequences = []
        seq_labels = []
        max_i = len(labels) - window
        for i in range(max_i):
            sequences.append(features[i:i+window])
            seq_labels.append(labels[i + window])
        return np.array(sequences), np.array(seq_labels)

    def predict_stock(self, ticker):
        data = Ticker(ticker).history(period="max")
        features = self.feature_engineer.calculate_features(data)
        latest_sequence = features[-30:].reshape(1, 30, -1)
        prediction = self.model.predict(latest_sequence)
        return self.class_labels[np.argmax(prediction)]

# Usage Example
if __name__ == "__main__":
    risk_params = {
        'risk_per_trade': 0.02,
        'stop_loss_multiplier': 1.5,
        'take_profit_multiplier': 3.0,
        'max_position_size': 0.15
    }
    
    predictor = UniversalSMCPredictor()
    
    try:
        predictor.load_model(r'C:\Roshan\nepse predictor\trained_models\model.keras')
    except:
        print("Training new model...")
        predictor.train_universal(['SPY', 'QQQ', 'AAPL'], '2010-01-01', '2023-01-01')
        predictor.save_model(r'C:\Roshan\nepse predictor\trained_models\model.keras')
    
    results = predictor.analyze_market(universe='sp500')
    
    if results:
        top_pick = results[0]['ticker']
        data = download(top_pick, period='1y')
        signals = predictor.generate_signals(data)
        positions = predictor.calculate_positions(data, signals)
        print(f"\nPosition Summary for {top_pick}:")
        print(positions[['position', 'entry_price', 'stop_loss', 'take_profit']].tail())