import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from yfinance import Ticker

# Enhanced SMC Feature Extractor
class SMCFeatureEngineer:
    def __init__(self, window=20):
        self.window = window
        self.scaler = StandardScaler()
        
    def calculate_features(self, df):
        # Basic Price Features
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.window).std()
        
        # SMC-Specific Features
        df['market_structure'] = self._calculate_market_structure(df)
        df['order_block_strength'] = self._calculate_order_blocks(df)
        df['liquidity_gradient'] = self._calculate_liquidity_zones(df)
        df['imbalance_ratio'] = self._calculate_imbalance_ratio(df)
        df['wyckoff_phase'] = self._calculate_wyckoff_phases(df)
        
        # Normalize features
        features = df[['market_structure', 'order_block_strength', 
                      'liquidity_gradient', 'imbalance_ratio', 
                      'wyckoff_phase', 'volatility']].dropna()
        return self.scaler.fit_transform(features)

    # Market Structure Analysis (unchanged)
    def _calculate_market_structure(self, df):
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
    def _calculate_order_blocks(self, df):
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
    def _calculate_liquidity_zones(self, df):
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
    def _calculate_imbalance_ratio(self, df):
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
    def _calculate_wyckoff_phases(self, df):
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
    def __init__(self):
        self.feature_engineer = SMCFeatureEngineer()
        self.model = self._build_model()
        self.class_labels = ['Buy', 'Sell', 'Hold']

    def _build_model(self):
        return SMCTransformer(num_features=6)

    def train_universal(self, stock_list, start_date, end_date):
        all_features = []
        all_labels = []
        
        for ticker in stock_list:
            data = Ticker(ticker).history(start=start_date, end=end_date)
            features = self.feature_engineer.calculate_features(data)
            labels = self._generate_labels(data)  # From previous SMC logic
            
            # Create sequences
            seq_features, seq_labels = self._create_sequences(features, labels)
            all_features.append(seq_features)
            all_labels.append(seq_labels)
            
        # Combine all data
        X = np.concatenate(all_features)
        y = np.concatenate(all_labels)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Compile and train
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
    # Initialize predictor
    smc_predictor = UniversalSMCPredictor()
    
    # Train on diverse universe of stocks
    training_stocks = ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOG', 'NVDA', 'SPY']
    smc_predictor.train_universal(training_stocks, 
                                 start_date='2015-01-01', 
                                 end_date='2023-01-01')

    # Predict for new stock
    prediction = smc_predictor.predict_stock('META')
    print(f"Predicted action for META: {prediction}")

    # Evaluate performance
    test_stocks = ['INTC', 'AMD', 'BTC-USD']
    for stock in test_stocks:
        pred = smc_predictor.predict_stock(stock)
        print(f"{stock} recommendation: {pred}")