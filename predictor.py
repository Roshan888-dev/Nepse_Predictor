from datetime import datetime
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from yfinance import Ticker, download

# Enhanced SMC Feature Extractor
class SMCFeatureEngineer:
    def __init__(self, window=20):
        self.window = window
        self.scaler = StandardScaler()
        
    def calculate_features(self, df):
        # Basic Price Features
        df = df.copy()
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.window).std()
        
        # Calculate SMC features with proper alignment
        df['market_structure'] = self._calculate_market_structure(df)
        df['order_block_strength'] = self._calculate_order_blocks(df)
        df['liquidity_gradient'] = self._calculate_liquidity_zones(df)
        df['imbalance_ratio'] = self._calculate_imbalance_ratio(df)
        df['wyckoff_phase'] = self._calculate_wyckoff_phases(df)
        
        # Normalize features with forward filling
        features = df[['market_structure', 'order_block_strength', 
                      'liquidity_gradient', 'imbalance_ratio', 
                      'wyckoff_phase', 'volatility']].ffill().dropna()
        
        return self.scaler.fit_transform(features)

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
        ob_strength = pd.Series(0.0, index=df.index)
        for i in range(2, len(df)):
            # Bullish Order Block
            if (df['Close'].iloc[i-2] < df['Open'].iloc[i-2] and
                df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and
                df['Close'].iloc[i] > df['Open'].iloc[i] and
                df['Volume'].iloc[i] > df['Volume'].iloc[i-1]*1.5):
                ob_strength.iloc[i] = 1.0  # Strength value
            # Bearish Order Block
            elif (df['Close'].iloc[i-2] > df['Open'].iloc[i-2] and
                df['Close'].iloc[i-1] > df['Open'].iloc[i-1] and
                df['Close'].iloc[i] < df['Open'].iloc[i] and
                df['Volume'].iloc[i] > df['Volume'].iloc[i-1]*1.5):
                ob_strength.iloc[i] = -1.0  # Strength value
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
            num_heads=4, key_dim=64)
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True))
        self.conv1d = tf.keras.layers.Conv1D(64, 3, activation='relu')
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='gelu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs):
        x = self.conv1d(inputs)
        x = self.lstm(x)
        x = self.attention(x, x)
        x = tf.reduce_mean(x, axis=1)
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
        return SMCTransformer(num_features=6)

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
    
    def generate_signals(self):  # Unchanged
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

    def analyze_market(self, universe='sp500', year_back=1):
        """Analyze entire market universe"""
        tickers = self._get_universe_tickers(universe)
        analysis_results = []
        
        for ticker in tickers:
            try:
                data = download(ticker, period=f'{year_back}y')
                if len(data) < 100:  # Skip illiquid stocks
                    continue
                    
                features = self.feature_engineer.calculate_features(data)
                signals = self.generate_signals(features)
                positions = self.calculate_positions(data, signals)
                
                # Generate analysis chart
                self.plot_analysis(ticker, data, signals, positions)
                
                # Save results
                analysis_results.append({
                    'ticker': ticker,
                    'current_signal': signals[-1],
                    'return_1m': data['Close'].pct_change(21)[-1],
                    'volatility': data['Close'].pct_change().std() * np.sqrt(252)
                })
                
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
        
        self.generate_market_report(analysis_results)
        return analysis_results

    def calculate_positions(self, data, signals):
        """Calculate position sizes and risk parameters"""
        positions = pd.DataFrame(index=data.index)
        positions['signal'] = signals
        positions['position'] = 0
        positions['entry_price'] = np.nan
        positions['stop_loss'] = np.nan
        positions['take_profit'] = np.nan
        
        in_position = False
        for i in range(len(positions)):
            if not in_position and positions.signal.iloc[i] == 'Long':
                entry_price = data['Close'].iloc[i]
                positions.entry_price.iloc[i] = entry_price
                positions.stop_loss.iloc[i] = entry_price * (1 - self.risk_params['stop_loss'])
                positions.take_profit.iloc[i] = entry_price * (1 + self.risk_params['take_profit'])
                positions.position.iloc[i] = self.risk_params['risk_per_trade']
                in_position = True
            elif in_position:
                if positions.signal.iloc[i] == 'Exit' or \
                   data['Low'].iloc[i] < positions.stop_loss.iloc[i-1] or \
                   data['High'].iloc[i] > positions.take_profit.iloc[i-1]:
                    positions.position.iloc[i] = 0
                    in_position = False
                else:
                    positions.position.iloc[i] = positions.position.iloc[i-1]
                    positions.entry_price.iloc[i] = positions.entry_price.iloc[i-1]
                    positions.stop_loss.iloc[i] = positions.stop_loss.iloc[i-1] * 1.01  # Trailing stop
                    positions.take_profit.iloc[i] = positions.take_profit.iloc[i-1] * 1.005
        return positions

    def plot_analysis(self, ticker, data, signals, positions):
        """Generate comprehensive analysis chart"""
        plt.figure(figsize=(20, 15))
        
        # Price Chart
        ax1 = plt.subplot(3, 1, 1)
        plt.plot(data['Close'], label='Price', color='black')
        plt.scatter(data.index[signals == 'Long'], 
                   data['Close'][signals == 'Long'], 
                   marker='^', color='g', s=100)
        plt.scatter(data.index[signals == 'Exit'], 
                   data['Close'][signals == 'Exit'], 
                   marker='v', color='r', s=100)
        plt.title(f'{ticker} Analysis - SMC Model')
        
        # Indicators
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        plt.plot(data['Close'].rolling(50).mean(), label='50 DMA')
        plt.plot(data['Close'].rolling(200).mean(), label='200 DMA')
        plt.legend()
        
        # Risk Management
        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        plt.plot(positions['position'], label='Position Size')
        plt.plot(positions['stop_loss'], label='Stop Loss', linestyle='--')
        plt.plot(positions['take_profit'], label='Take Profit', linestyle='--')
        plt.legend()
        
        # Save chart
        os.makedirs('analysis_charts', exist_ok=True)
        plt.savefig(f'analysis_charts/{ticker}_analysis.png')
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
        """Generate summary market report"""
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
    def train_universal(self, stock_list, start_date, end_date):
        all_features = []
        all_labels = []
        
        for ticker in stock_list:
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
    smc = UniversalSMCPredictor()
    
    # Load pre-trained model or train new
    try:
        smc.load_model()
    except:
        smc.train_universal(['SPY', 'QQQ'], '2010-01-01', '2023-01-01')
        smc.save_model()
    
    # Full market analysis
    results = smc.analyze_market(universe='sp500')
    
    # Generate positions for top candidate
    top_pick = results[0]['ticker']
    data = download(top_pick, period='1y')
    features = smc.feature_engineer.calculate_features(data)
    signals = smc.generate_signals(features)
    positions = smc.calculate_positions(data, signals)
    
    print(f"\nRecommended Position for {top_pick}:")
    print(positions.tail(10))