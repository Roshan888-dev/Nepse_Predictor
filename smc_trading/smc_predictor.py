import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from typing import List, Dict
from config import TRADING_CONFIG, PATHS, DATA_CONFIG, MODEL_CONFIG
from data_processor import UniversalDataProcessor, MultiTimeFrameAnalyzer
from feature_engineer import SMCFeatureEngineer
from model import create_multi_input_model
from utils import download_market_data, generate_report

class UniversalSMCPredictor:
    def __init__(self):
        self.data_processor = UniversalDataProcessor()
        self.feature_engineer = SMCFeatureEngineer()
        self.analyzer = MultiTimeFrameAnalyzer()
        self.model = create_multi_input_model()
        self._setup_directories()
        
    def _setup_directories(self):
        """Ensure all required directories exist."""
        for path in PATHS.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def train(self, tickers: List[str], start: str, end: str):
        """Train the model on a list of tickers."""
        X, y = [], []
        
        for ticker in tickers:
            try:
                # Download data with start/end dates
                data = download_market_data(ticker, start=start, end=end)
                
                # Validate data
                if data.empty or len(data) < 100:
                    print(f"Skipping {ticker} - insufficient data")
                    continue
                    
                # Process data
                processed = self.data_processor.process(data)
                features = self.feature_engineer.calculate_features(processed)
                
                # Ensure features are valid
                if len(features) < DATA_CONFIG['sequence_lengths']['daily']:
                    print(f"Skipping {ticker} - not enough features")
                    continue
                    
                labels = self._generate_labels(processed)
                seq_features, seq_labels = self._create_multi_sequences(features, labels)
                
                X.append(seq_features)
                y.append(seq_labels)
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
        
        # Check if we have training data
        if not X:
            raise ValueError("No valid training data found. Check input tickers and date range.")
        
        X = np.concatenate(X)
        y = np.concatenate(y)
        
        # Split data only if we have samples
        if len(X) == 0:
            raise ValueError("No valid sequences generated. Check feature engineering.")
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=MODEL_CONFIG['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model.fit(
            self._format_multi_input(X_train), y_train,
            validation_data=(self._format_multi_input(X_test), y_test),
            epochs=100, batch_size=256,
            callbacks=[tf.keras.callbacks.ModelCheckpoint(str(PATHS['models'] / 'smc_model.keras'), save_best_only=True)]
        )
    
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Generate predictions for the given data.
        
        Args:
            data (pd.DataFrame): Input data with OHLCV columns.
        
        Returns:
            Dict: Predictions with probabilities for each class.
        """
        processed = self.data_processor.process(data)
        features = self.feature_engineer.calculate_features(processed)
        sequences = self._create_multi_sequences(features, np.zeros(len(features)))[0]  # Dummy labels
        predictions = self.model.predict(self._format_multi_input(sequences))
        return {
            'Long': predictions[:, 0],
            'Neutral': predictions[:, 1],
            'Exit': predictions[:, 2]
        }
    
    def analyze_market(self, universe: str = 'sp500') -> pd.DataFrame:
        """
        Analyze the entire market for trading opportunities.
        
        Args:
            universe (str): Market universe to analyze ('sp500', 'nasdaq100', etc.).
        
        Returns:
            pd.DataFrame: Analysis results with signals and metrics.
        """
        tickers = self._get_universe_tickers(universe)
        results = []
        
        for ticker in tickers:
            try:
                data = download_market_data(ticker, period='1y')
                if len(data) < 100:
                    continue
                
                signals = self.predict(data)
                positions = self.calculate_positions(data, signals)
                results.append({
                    'ticker': ticker,
                    'signal': signals['Long'][-1],  # Latest signal
                    'return_1m': data['Close'].pct_change(21).iloc[-1],
                    'volatility': data['Close'].pct_change().std() * np.sqrt(252)
                })
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
        
        results_df = pd.DataFrame(results)
        generate_report(results_df)
        return results_df
    
    def calculate_positions(self, data: pd.DataFrame, signals: Dict) -> pd.DataFrame:
        """
        Calculate position sizes and risk parameters.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns.
            signals (Dict): Model predictions with probabilities.
        
        Returns:
            pd.DataFrame: Position details including stop loss and take profit.
        """
        positions = pd.DataFrame(index=data.index)
        positions['signal'] = np.argmax(np.array([signals['Long'], signals['Neutral'], signals['Exit']]), axis=0)
        positions['position'] = 0.0
        positions['stop_loss'] = np.nan
        positions['take_profit'] = np.nan
        
        in_position = False
        for i in range(1, len(positions)):
            if not in_position and positions['signal'].iloc[i] == 0:  # Long signal
                entry_price = data['Close'].iloc[i]
                atr = data['ATR'].iloc[i]
                positions.at[i, 'position'] = TRADING_CONFIG['risk_per_trade'] / (entry_price * TRADING_CONFIG['stop_loss_multiplier'] * atr)
                positions.at[i, 'stop_loss'] = entry_price - TRADING_CONFIG['stop_loss_multiplier'] * atr
                positions.at[i, 'take_profit'] = entry_price + TRADING_CONFIG['take_profit_multiplier'] * atr
                in_position = True
            elif in_position:
                if positions['signal'].iloc[i] == 2 or data['Low'].iloc[i] < positions['stop_loss'].iloc[i-1]:
                    in_position = False
                else:
                    positions.at[i, 'position'] = positions.at[i-1, 'position']
                    positions.at[i, 'stop_loss'] = max(positions.at[i-1, 'stop_loss'], data['Close'].iloc[i] - TRADING_CONFIG['stop_loss_multiplier'] * data['ATR'].iloc[i])
        
        return positions
    
    def _generate_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate target labels based on future price movements.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV columns.
        
        Returns:
            np.ndarray: Array of labels (0: Long, 1: Neutral, 2: Exit).
        """
        future_returns = data['Close'].pct_change(3).shift(-3)
        labels = np.where(
            future_returns > 0.02, 0,  # Long
            np.where(future_returns < -0.02, 2, 1)  # Exit or Neutral
        )
        return labels[:len(data) - 3]  # Remove last 3 NaN values
    
    def _create_multi_sequences(self, features: np.ndarray, labels: np.ndarray) -> tuple:
        """Create multi-timeframe sequences with validation."""
        sequences = []
        seq_labels = []
        window = DATA_CONFIG['sequence_lengths']['daily']
        
        # Ensure we have enough data
        if len(features) < window:
            return np.array([]), np.array([])
        
        for i in range(len(features) - window):
            try:
                daily = features[i:i+window]
                weekly = features[i//5:i//5+DATA_CONFIG['sequence_lengths']['weekly']]
                monthly = features[i//21:i//21+DATA_CONFIG['sequence_lengths']['monthly']]
                
                # Validate sequence lengths
                if (len(weekly) == DATA_CONFIG['sequence_lengths']['weekly'] and 
                    len(monthly) == DATA_CONFIG['sequence_lengths']['monthly']):
                    sequences.append({
                        'daily': daily,
                        'weekly': weekly,
                        'monthly': monthly
                    })
                    seq_labels.append(labels[i+window])
                    
            except IndexError:
                continue
                
        return np.array(sequences), np.array(seq_labels)
    
    def _format_multi_input(self, sequences: np.ndarray) -> Dict:
        """
        Format sequences into model input format.
        
        Args:
            sequences (np.ndarray): Input sequences.
        
        Returns:
            Dict: Formatted inputs for the model.
        """
        return {
            'daily': np.array([x['daily'] for x in sequences]),
            'weekly': np.array([x['weekly'] for x in sequences]),
            'monthly': np.array([x['monthly'] for x in sequences]),
            'market_structure': np.array([x['market_structure'] for x in sequences])
        }
    
    def _get_universe_tickers(self, universe: str) -> List[str]:
        """
        Get tickers for a specific market universe.
        
        Args:
            universe (str): Market universe ('sp500', 'nasdaq100', etc.).
        
        Returns:
            List[str]: List of tickers.
        """
        if universe == 'sp500':
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            return table[0]['Symbol'].tolist()
        elif universe == 'nasdaq100':
            return pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]['Ticker'].tolist()
        else:
            raise ValueError(f"Unsupported universe: {universe}")