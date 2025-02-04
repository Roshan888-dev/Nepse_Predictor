"""
Upgraded Universal SMC Predictor Model with Full SMC Concept Integration,
Multi-Timeframe Analysis, Enhanced Position Planning, and Accuracy Metrics

This module implements an advanced SMC-based predictor that:
  1. Computes SMC features including:
     - Market Structure: Trend, BOS, and ChoCH.
     - Order Blocks: Institutional order activity.
     - Liquidity: Areas of concentrated orders.
     - Imbalances: Fair value gaps.
     - Supply/Demand Zones: Approximate support/resistance.
     - Volatility.
  2. Generates trade signals in the form of "Buy", "Sell", or "Neutral"
     using confluence from multiple SMC concepts.
  3. Applies multi-timeframe confirmation (daily and weekly).
  4. Plans positions (for both Buy and Sell) with risk management.
  5. Trains a transformer-based neural network model and calculates accuracy metrics.

Author: [Your Name]
Date: [Current Date]
"""

from datetime import datetime
import os
import logging
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from yfinance import Ticker, download
import matplotlib.dates as mdates
import matplotlib

# Use a non-interactive backend for matplotlib
matplotlib.use('Agg')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


# =============================================================================
# SMC Feature Engineer
# =============================================================================
class SMCFeatureEngineer:
    """
    Compute technical and SMC-specific features from OHLCV data.
    
    Features include:
      - Market Structure: Trend, Breaks of Structure (BOS) and Change of Character (ChoCH).
      - Order Block Strength: Areas of institutional order activity.
      - Liquidity Gradient: Using swing highs/lows.
      - Imbalance Ratio: Fair value gap detection.
      - Wyckoff Phase: A proxy for accumulation/distribution.
      - Supply/Demand Zones: Approximated by proximity to recent rolling min/max.
      - Volatility: Rolling standard deviation of returns.
      
    The features are then scaled for model input.
    """
    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.scaler = StandardScaler()
        
    def calculate_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate and return a feature matrix from OHLCV data.
        
        Args:
            df (pd.DataFrame): Must include columns ['Open', 'High', 'Low', 'Close', 'Volume'].
            
        Returns:
            np.ndarray: Scaled feature matrix with 7 features.
        """
        df = df.copy()
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.window).std()
        
        # SMC Components
        df['market_structure'] = self._calculate_market_structure(df)
        df['order_block_strength'] = self._calculate_order_blocks(df)
        df['liquidity_gradient'] = self._calculate_liquidity_zones(df)
        df['imbalance_ratio'] = self._calculate_imbalance_ratio(df)
        df['wyckoff_phase'] = self._calculate_wyckoff_phases(df)
        df['supply_demand'] = self._calculate_supply_demand_zones(df)
        
        # Select features; fill forward and drop any NaNs
        features = df[['market_structure', 'order_block_strength', 
                       'liquidity_gradient', 'imbalance_ratio', 
                       'wyckoff_phase', 'volatility', 'supply_demand']].ffill().dropna()
        
        return self.scaler.fit_transform(features)

    def _calculate_market_structure(self, df: pd.DataFrame) -> pd.Series:
        """
        Determine the market structure:
         - 1: Bullish break of structure (BOS)
         - -1: Bearish BOS
         - 2: Bullish change of character (ChoCH)
         - -2: Bearish ChoCH
         - 0: Neutral
        
        Uses recent highs and lows.
        """
        structure = pd.Series(0, index=df.index, dtype=int)
        highs = df['High']
        lows = df['Low']
        for i in range(2, len(df) - 2):
            current_high = highs.iloc[i]
            prev1_high = highs.iloc[i - 1]
            prev2_high = highs.iloc[i - 2]
            current_low = lows.iloc[i]
            prev1_low = lows.iloc[i - 1]
            prev2_low = lows.iloc[i - 2]
            
            # BOS detection
            if current_high > prev1_high and current_high > prev2_high:
                structure.iloc[i] = 1
            elif current_low < prev1_low and current_low < prev2_low:
                structure.iloc[i] = -1
            # ChoCH detection
            if (current_high < prev1_high and prev1_high > prev2_high and
                current_low > prev1_low and prev1_low < prev2_low):
                structure.iloc[i] = 2
            elif (current_low > prev1_low and prev1_low < prev2_low and
                  current_high < prev1_high and prev1_high > prev2_high):
                structure.iloc[i] = -2
        return structure
    
    def _calculate_order_blocks(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute order block strength using a simple heuristic.
        
        Returns 1 for bullish order blocks and -1 for bearish.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        ob_strength = pd.Series(0.0, index=df.index)
        bearish = ((df['Close'].shift(2) > df['Open'].shift(2)) &
                   (df['Close'].shift(1) > df['Open'].shift(1)) &
                   (df['Close'] < df['Open']) &
                   (df['Volume'] > df['Volume'].shift(1) * 1.5))
        bullish = ((df['Close'].shift(2) < df['Open'].shift(2)) &
                   (df['Close'].shift(1) < df['Open'].shift(1)) &
                   (df['Close'] > df['Open']) &
                   (df['Volume'] > df['Volume'].shift(1) * 1.5))
        ob_strength = ob_strength.mask(bullish, 1.0)
        ob_strength = ob_strength.mask(bearish, -1.0)
        return ob_strength

    def _calculate_liquidity_zones(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the liquidity gradient by identifying swing highs and lows.
        A rolling mean smooths the gradient.
        """
        gradient = pd.Series(0.0, index=df.index)
        swing_highs = df['High'][(df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))]
        swing_lows = df['Low'][(df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))]
        gradient.loc[swing_highs.index] = 1.0
        gradient.loc[swing_lows.index] = -1.0
        return gradient.rolling(5).mean().fillna(0)

    def _calculate_imbalance_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect imbalances (fair value gaps) by comparing current and next bars.
        Returns 1 for upward gaps, -1 for downward gaps.
        """
        imbalance = pd.Series(0.0, index=df.index)
        for i in range(1, len(df) - 1):
            if df['High'].iloc[i] < df['Low'].iloc[i + 1]:
                imbalance.iloc[i] = 1.0
            elif df['Low'].iloc[i] > df['High'].iloc[i + 1]:
                imbalance.iloc[i] = -1.0
        return imbalance

    def _calculate_wyckoff_phases(self, df: pd.DataFrame) -> pd.Series:
        """
        Determine accumulation/distribution phases (a simplified Wyckoff phase indicator).
        Returns 1 for accumulation, -1 for distribution.
        """
        wyckoff = pd.Series(0, index=df.index, dtype=int)
        vwap = df['VWAP'] if 'VWAP' in df.columns else df['Close']
        for i in range(4, len(df) - 4):
            if (df['Volume'].iloc[i] > df['Volume'].iloc[i - 1] and
                df['Close'].iloc[i] > vwap.iloc[i] and
                df['Low'].iloc[i] > df['Low'].iloc[i - 2]):
                wyckoff.iloc[i] = 1
            elif (df['Volume'].iloc[i] > df['Volume'].iloc[i - 1] and
                  df['Close'].iloc[i] < vwap.iloc[i] and
                  df['High'].iloc[i] < df['High'].iloc[i - 2]):
                wyckoff.iloc[i] = -1
        return wyckoff

    def _calculate_supply_demand_zones(self, df: pd.DataFrame) -> pd.Series:
        """
        Approximate supply and demand zones.
        
        A simple heuristic: if the Close is within 1% of the recent rolling min, mark as demand (1);
        if within 1% of the recent rolling max, mark as supply (-1); otherwise 0.
        """
        window = self.window
        rolling_min = df['Low'].rolling(window).min()
        rolling_max = df['High'].rolling(window).max()
        sd = pd.Series(0.0, index=df.index)
        demand_zone = abs(df['Close'] - rolling_min) / rolling_min < 0.01
        supply_zone = abs(df['Close'] - rolling_max) / rolling_max < 0.01
        sd[demand_zone] = 1.0
        sd[supply_zone] = -1.0
        return sd


# =============================================================================
# SMC Neural Network Architecture
# =============================================================================
class SMCTransformer(tf.keras.Model):
    """
    Transformer-based neural network model for SMC prediction.
    """
    def __init__(self, num_features: int, num_classes: int = 3):
        """
        Initialize the SMCTransformer model.
        
        Args:
            num_features (int): Number of features per time step.
            num_classes (int): Number of target classes.
        """
        super(SMCTransformer, self).__init__()
        self.conv1d = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='gelu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass.
        """
        x = self.conv1d(inputs)
        x = self.lstm(x)
        x = self.attention(x, x)
        x = self.global_pool(x)
        return self.classifier(x)


# =============================================================================
# Universal SMC Predictor
# =============================================================================
class UniversalSMCPredictor:
    """
    Universal SMC predictor that integrates advanced SMC calculations, multi-timeframe analysis,
    enhanced Buy/Sell (position planning) with risk management, and accuracy metrics.
    """
    def __init__(self,
                 risk_per_trade: float = 0.01,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.10) -> None:
        self.feature_engineer = SMCFeatureEngineer()
        self.model = self._build_model()
        # Label mapping: 0 = Buy, 1 = Sell, 2 = Neutral.
        self.class_labels = ['Buy', 'Sell', 'Neutral']
        self.risk_params = {
            'risk_per_trade': risk_per_trade,
            'stop_loss': stop_loss_pct,
            'take_profit': take_profit_pct
        }
        
    def _build_model(self) -> SMCTransformer:
        """
        Build the transformer model.
        """
        # Now using 7 features as computed by our feature engineer.
        return SMCTransformer(num_features=7)

    def save_model(self, path: str = r'./smc_model.keras') -> None:
        """
        Save the trained model.
        """
        self.model.save(path)
        logging.info(f"Model saved to {path}")

    def load_model(self, path: str = './smc_model.keras') -> None:
        """
        Load a pre-trained model.
        """
        try:
            self.model = tf.keras.models.load_model(path, custom_objects={"SMCTransformer": SMCTransformer})
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def _generate_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate labels based on future returns over a 3-day horizon.
          - ret > 2%: Buy (0)
          - ret < -2%: Sell (1)
          - Otherwise: Neutral (2)
        """
        future_returns = data['Close'].pct_change(3).shift(-3)
        labels = []
        for ret in future_returns:
            if pd.isna(ret):
                continue
            if ret > 0.02:
                labels.append(0)
            elif ret < -0.02:
                labels.append(1)
            else:
                labels.append(2)
        return np.array(labels)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trade signals on the base timeframe using SMC features.
        Incorporates confluence from order blocks, market structure, liquidity,
        imbalances, Wyckoff phase, and supply/demand.
        
        Returns:
            pd.Series: "Buy", "Sell", or "Neutral" for each time step.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        df = data.copy()
        df['ob_strength'] = self.feature_engineer._calculate_order_blocks(df)
        df['market_structure'] = self.feature_engineer._calculate_market_structure(df)
        df['liquidity_gradient'] = self.feature_engineer._calculate_liquidity_zones(df)
        df['imbalance_ratio'] = self.feature_engineer._calculate_imbalance_ratio(df)
        df['wyckoff_phase'] = self.feature_engineer._calculate_wyckoff_phases(df)
        df['supply_demand'] = self.feature_engineer._calculate_supply_demand_zones(df)
        
        signals = []
        for _, row in df.iterrows():
            # Default signal is Neutral.
            signal = 'Neutral'
            # A confluence rule: if at least 4 of 6 conditions favor one direction.
            buy_conditions = [
                row['ob_strength'] > 0.8,
                row['market_structure'] in [1, 2],
                row['liquidity_gradient'] < -0.7,
                row['imbalance_ratio'] > 0.7,
                row['wyckoff_phase'] == 1,
                row['supply_demand'] == 1
            ]
            sell_conditions = [
                row['ob_strength'] < -0.8,
                row['market_structure'] in [-1, -2],
                row['liquidity_gradient'] > 0.7,
                row['imbalance_ratio'] < -0.7,
                row['wyckoff_phase'] == -1,
                row['supply_demand'] == -1
            ]
            if sum(buy_conditions) >= 4:
                signal = 'Buy'
            elif sum(sell_conditions) >= 4:
                signal = 'Sell'
            signals.append(signal)
        return pd.Series(signals, index=df.index, name='signals')

    def multi_timeframe_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals on multiple timeframes (daily and weekly) and return a consensus.
        """
        daily = self.generate_signals(data)
        weekly_data = data.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        weekly = self.generate_signals(weekly_data)
        weekly_daily = weekly.reindex(data.index, method='ffill')
        consensus = []
        for d, s1, s2 in zip(data.index, daily, weekly_daily):
            if s1 == s2 and s1 in ['Buy', 'Sell']:
                consensus.append(s1)
            else:
                consensus.append('Neutral')
        return pd.Series(consensus, index=data.index, name='consensus_signals')

    def plan_positions(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """
        Plan positions for both Buy and Sell trades based on the consensus signals.
        Calculates entry prices, stop-loss, and take-profit levels.
        """
        positions = pd.DataFrame(index=data.index)
        positions['signal'] = signals
        positions['position'] = 0.0
        positions['entry_price'] = np.nan
        positions['stop_loss'] = np.nan
        positions['take_profit'] = np.nan
        
        in_buy = False
        in_sell = False
        for i in range(len(positions)):
            current = positions.index[i]
            # Buy position planning
            if not in_buy and positions['signal'].iat[i] == 'Buy':
                entry = data['Close'].iat[i]
                positions.at[current, 'entry_price'] = entry
                positions.at[current, 'stop_loss'] = entry * (1 - self.risk_params['stop_loss'])
                positions.at[current, 'take_profit'] = entry * (1 + self.risk_params['take_profit'])
                positions.at[current, 'position'] = self.risk_params['risk_per_trade']
                in_buy = True
            elif in_buy:
                prev = positions.index[i - 1]
                if (positions['signal'].iat[i] != 'Buy' or
                    data['Low'].iat[i] < positions.at[prev, 'stop_loss'] or
                    data['High'].iat[i] > positions.at[prev, 'take_profit']):
                    positions.at[current, 'position'] = 0.0
                    in_buy = False
                else:
                    positions.at[current, 'position'] = positions.at[prev, 'position']
                    positions.at[current, 'entry_price'] = positions.at[prev, 'entry_price']
                    positions.at[current, 'stop_loss'] = positions.at[prev, 'stop_loss'] * 1.01
                    positions.at[current, 'take_profit'] = positions.at[prev, 'take_profit'] * 1.005
            # Sell position planning (inverted logic)
            if not in_sell and positions['signal'].iat[i] == 'Sell':
                entry = data['Close'].iat[i]
                positions.at[current, 'entry_price'] = entry
                positions.at[current, 'stop_loss'] = entry * (1 + self.risk_params['stop_loss'])
                positions.at[current, 'take_profit'] = entry * (1 - self.risk_params['take_profit'])
                positions.at[current, 'position'] = -self.risk_params['risk_per_trade']
                in_sell = True
            elif in_sell:
                prev = positions.index[i - 1]
                if (positions['signal'].iat[i] != 'Sell' or
                    data['High'].iat[i] > positions.at[prev, 'stop_loss'] or
                    data['Low'].iat[i] < positions.at[prev, 'take_profit']):
                    positions.at[current, 'position'] = 0.0
                    in_sell = False
                else:
                    positions.at[current, 'position'] = positions.at[prev, 'position']
                    positions.at[current, 'entry_price'] = positions.at[prev, 'entry_price']
                    positions.at[current, 'stop_loss'] = positions.at[prev, 'stop_loss'] * 1.01
                    positions.at[current, 'take_profit'] = positions.at[prev, 'take_profit'] * 1.005
        return positions

    def plot_analysis(self, ticker: str, data: pd.DataFrame, signals: pd.Series, positions: pd.DataFrame) -> None:
        """
        Plot price, signals, and risk management levels.
        """
        fig, axs = plt.subplots(2, 1, figsize=(20, 10), constrained_layout=True)
        ax1 = axs[0]
        ax1.plot(data.index, data['Close'], '-', label='Price', color='blue', linewidth=1.5)
        buy_signals = signals == 'Buy'
        sell_signals = signals == 'Sell'
        if buy_signals.any():
            ax1.plot(data.loc[buy_signals].index, data.loc[buy_signals, 'Close'],
                     '^', color='green', ms=10, label='Buy Signal')
        if sell_signals.any():
            ax1.plot(data.loc[sell_signals].index, data.loc[sell_signals, 'Close'],
                     'v', color='red', ms=10, label='Sell Signal')
        if positions['stop_loss'].notna().any():
            ax1.plot(positions.index, positions['stop_loss'],
                     '--', color='maroon', label='Stop Loss')
        if positions['take_profit'].notna().any():
            ax1.plot(positions.index, positions['take_profit'],
                     '--', color='darkblue', label='Take Profit')
        ax1.set_title(f'{ticker} Analysis - SMC Model', fontsize=16, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        for idx, signal in signals.items():
            if signal == 'Buy':
                ax1.annotate('Buy', xy=(idx, data.loc[idx, 'Close']),
                             xytext=(10, 20), textcoords='offset points',
                             arrowprops=dict(arrowstyle='->', color='green'))
            elif signal == 'Sell':
                ax1.annotate('Sell', xy=(idx, data.loc[idx, 'Close']),
                             xytext=(10, -20), textcoords='offset points',
                             arrowprops=dict(arrowstyle='->', color='red'))
        ax2 = axs[1]
        ma50 = data['Close'].rolling(50).mean()
        ma200 = data['Close'].rolling(200).mean()
        ax2.plot(data.index, ma50, '-', label='50 DMA', color='orange', linewidth=1.5)
        ax2.plot(data.index, ma200, '-', label='200 DMA', color='purple', linewidth=1.5)
        ax2.fill_between(data.index, data['Close'].rolling(50).min(),
                         data['Close'].rolling(50).max(), color='gray', alpha=0.2, label='50 Period Range')
        ax2.set_title('Trend Analysis', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        os.makedirs('analysis_charts', exist_ok=True)
        chart_path = f'analysis_charts/{ticker}_analysis.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Analysis chart saved to {chart_path}")

    def _get_universe_tickers(self, universe: str = 'sp500') -> list:
        """
        Retrieve tickers from a specified universe.
        """
        if universe.lower() == 'sp500':
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            return table[0]['Symbol'].tolist()
        elif universe.lower() == 'nasdaq100':
            table = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
            return table[4]['Ticker'].tolist()
        else:
            raise ValueError("Unsupported universe")

    def generate_market_report(self, results: list) -> None:
        """
        Generate and save a market analysis report.
        """
        df = pd.DataFrame(results)
        df = df[df['current_signal'].isin(['Buy', 'Sell'])].sort_values('return_1m', ascending=False)
        report = f"""
        SMC Market Analysis Report - {datetime.today().date()}
        ------------------------------------------------
        Total Opportunities Found: {len(df)}
        Top 5 Candidates:
        {df.head(5).to_string(index=False)}
        
        Market Statistics:
        Average 1M Return: {df.return_1m.mean():.2%}
        Average Volatility: {df.volatility.mean():.2%}
        """
        with open('market_report.txt', 'w') as f:
            f.write(report)
        logging.info("Market report generated:")
        logging.info(report)

    def analyze_market(self, universe: str = 'sp500', year_back: int = 1, batch_size: int = 50) -> list:
        """
        Analyze a market universe in batches.
        """
        tickers = self._get_universe_tickers(universe)
        analysis_results = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            for ticker in batch:
                try:
                    data = download(ticker, period=f'{year_back}y')
                    if data.empty or len(data) < 100:
                        continue
                    df = data.copy()
                    consensus = self.multi_timeframe_signals(df)
                    positions = self.plan_positions(df, consensus)
                    self.plot_analysis(ticker, df, consensus, positions)
                    analysis_results.append({
                        'ticker': ticker,
                        'current_signal': consensus.iloc[-1],
                        'return_1m': df['Close'].pct_change(21).iloc[-1],
                        'volatility': df['Close'].pct_change().std() * np.sqrt(252)
                    })
                except Exception as e:
                    logging.error(f"Error processing {ticker}: {e}")
            plt.close('all')
        self.generate_market_report(analysis_results)
        return analysis_results

    def _create_sequences(self, features: np.ndarray, labels: np.ndarray, window: int = 30) -> tuple:
        """
        Create time-series sequences for training.
        """
        sequences, seq_labels = [], []
        for i in range(len(features) - window):
            sequences.append(features[i:i + window])
            seq_labels.append(labels[i + window])
        return np.array(sequences), np.array(seq_labels)

    def train_universal(self, stock_list: list, start_date: str, end_date: str, epochs: int = 50, batch_size: int = 64) -> None:
        """
        Train the SMC model using historical stock data.
        """
        all_features, all_labels = [], []
        for ticker in stock_list:
            try:
                logging.info(f"Fetching data for {ticker}")
                raw = Ticker(ticker).history(start=start_date, end=end_date)
                if raw.empty or len(raw) < 100:
                    continue
                feats = self.feature_engineer.calculate_features(raw)
                labs = self._generate_labels(raw)
                mlen = min(len(feats), len(labs))
                feats, labs = feats[:mlen], labs[:mlen]
                seq_feats, seq_labs = self._create_sequences(feats, labs)
                all_features.append(seq_feats)
                all_labels.append(seq_labs)
            except Exception as e:
                logging.error(f"Error processing {ticker}: {e}")
        if not all_features:
            logging.error("No data available for training.")
            return
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        logging.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        self.model.compile(optimizer='adamax',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(X_train, y_train,
                       validation_data=(X_test, y_test),
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=callbacks)
        logging.info("Training completed.")
        self.evaluate_model(X_test, y_test)

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Evaluate the model and log accuracy metrics.
        """
        preds = np.argmax(self.model.predict(X_test), axis=1)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, preds)
        logging.info("Evaluation Metrics:")
        logging.info(f"Accuracy: {acc:.4f}")
        logging.info(f"Precision: {prec:.4f}")
        logging.info(f"Recall: {rec:.4f}")
        logging.info(f"F1-Score: {f1:.4f}")
        logging.info(f"Confusion Matrix:\n{cm}")

    def predict_stock(self, ticker: str) -> str:
        """
        Predict a trading signal for a given ticker using the latest data.
        """
        data = download(ticker, period="max")
        if data.empty:
            raise ValueError(f"No data returned for ticker {ticker}")
        feats = self.feature_engineer.calculate_features(data)
        if len(feats) < 30:
            raise ValueError("Not enough data to form a valid sequence for prediction.")
        latest_seq = feats[-30:].reshape(1, 30, -1)
        prediction = self.model.predict(latest_seq)
        return self.class_labels[np.argmax(prediction)]


# =============================================================================
# Usage Example
# =============================================================================
if __name__ == "__main__":
    smc = UniversalSMCPredictor()
    
    # Attempt to load an existing model; otherwise, train a new one.
    try:
        smc.load_model(path='./smc_model.keras')
    except Exception:
        logging.info("No pre-trained model found. Starting training...")
        smc.train_universal(['SPY', 'QQQ'], '2010-01-01', '2023-01-01', epochs=50, batch_size=64)
        smc.save_model(path='./smc_model.keras')
    
    # Run full market analysis
    logging.info("Starting market analysis...")
    results = smc.analyze_market(universe='sp500', year_back=1, batch_size=50)
    
    # Generate trade signals and planned positions for the top candidate
    if results:
        top_ticker = results[0]['ticker']
        logging.info(f"Top candidate ticker: {top_ticker}")
        data = download(top_ticker, period='1y')
        if data.empty:
            logging.error(f"Failed to download data for {top_ticker}")
        else:
            consensus = smc.multi_timeframe_signals(data)
            positions = smc.plan_positions(data, consensus)
            logging.info(f"Recommended Positions for {top_ticker}:")
            logging.info(positions.tail(10))
    else:
        logging.error("Market analysis returned no results.")
