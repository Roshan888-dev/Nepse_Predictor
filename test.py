import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from yfinance import download
import matplotlib.pyplot as plt

from predictor import SMCFeatureEngineer

class UniversalSMCPredictor:
    def __init__(self, risk_per_trade=0.01, stop_loss_pct=0.05, take_profit_pct=0.10):
        self.feature_engineer = SMCFeatureEngineer()
        self.model = tf.keras.models.load_model('C:/Roshan/nepse predictor/model.keras') # Path to saved model
        self.class_labels = ['Long', 'Neutral', 'Exit']
        self.risk_params = {
            'risk_per_trade': risk_per_trade,
            'stop_loss': stop_loss_pct,
            'take_profit': take_profit_pct
        }

    def _load_model(self, path):
        """Load the pre-trained model from the specified path."""
        model = tf.keras.models.load_model(path)
        print("Model loaded successfully from:", path)
        return model

    def load_and_predict(self, ticker, period='1y', window=20):
        """Load model and predict based on new data"""
        # Download the historical data for the ticker
        data = download(ticker, period=period)
        
        # Check if data is sufficient
        if len(data) < window:
            raise ValueError(f"Not enough data for {ticker} for the given period.")
        
        # Calculate features using the feature engineer
        features = self.feature_engineer.calculate_features(data)

        # Prepare data for prediction
        X_new = features.reshape(1, -1)  # Reshape data to fit the model input

        # Predict using the pre-trained model
        prediction = self.model.predict(X_new)

        # Map the model output to class labels
        predicted_label = np.argmax(prediction, axis=1)

        return self.class_labels[predicted_label[0]]

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
            if any([ob > 0.8, ms in [1, 2], liq < -0.7, imb > 0.7, wyk == 1]):
                signal = 'Long'
            elif any([ob < -0.8, ms in [-1, -2], liq > 0.7, imb < -0.7, wyk == -1]):
                signal = 'Exit'
                
            signals.append(signal)
        
        return pd.Series(signals, index=df.index, name='signals')

    def plot_analysis(self, ticker, data, signals, positions):
        """Plot analysis of the stock's movements along with the trading signals"""
        plt.figure(figsize=(12, 6))
        
        # Plot the close price
        plt.plot(data['Close'], label='Close Price', color='blue', alpha=0.7)
        
        # Plot Buy signals
        plt.plot(data[signals == 'Long'].index, 
                 data[signals == 'Long']['Close'], 
                 '^', markersize=10, color='green', label='Buy Signal')
        
        # Plot Exit signals
        plt.plot(data[signals == 'Exit'].index, 
                 data[signals == 'Exit']['Close'], 
                 'v', markersize=10, color='red', label='Exit Signal')
        
        plt.title(f"{ticker} Trading Signals")
        plt.legend()
        plt.show()

    def evaluate_model(self, true_labels, predicted_labels):
        """Evaluate the model's predictions using industry-standard metrics."""
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
        recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
        f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def analyze_market(self, universe='sp500', year_back=1, batch_size=50):
        """Analyze entire market universe with proper DataFrame handling"""
        tickers = self._get_universe_tickers(universe)
        analysis_results = []
        
        # Process tickers in batches to avoid memory issues
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            for ticker in batch_tickers:
                try:
                    data = download(ticker, period=f'{year_back}y')
                    if len(data) < 100:
                        continue
                        
                    # Ensure we're working with proper DataFrame
                    df = data.copy()
                    features = self.feature_engineer.calculate_features(df)
                    signals = self.generate_signals(df)  # Pass DataFrame instead of features array
                    positions = self.calculate_positions(df, signals)
                    
                    # Generate analysis chart with enhanced visuals
                    self.plot_analysis(ticker, df, signals, positions)
                    
                    # Safe indexing using iloc
                    analysis_results.append({
                        'ticker': ticker,
                        'current_signal': signals.iloc[-1],  # Use iloc for positional access
                        'return_1m': df['Close'].pct_change(21).iloc[-1],
                        'volatility': df['Close'].pct_change().std() * np.sqrt(252)
                    })
                    
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
            
            # Clear memory after each batch
            plt.close('all')
        
        self.generate_market_report(analysis_results)
        return analysis_results

    def _get_universe_tickers(self, universe):
        """Return list of tickers based on the universe"""
        # Here it returns tickers from the S&P500 for demonstration
        if universe == 'sp500':
            return ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']  # Example tickers (S&P 500)
        return []

# Example usage
if __name__ == "__main__":
    predictor = UniversalSMCPredictor()

    # Define the list of tickers (for example, top S&P 500 companies)
    tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

    true_labels = []  # Placeholder for true labels (to be filled with actual labels)
    predicted_labels = []  # Placeholder for predicted labels

    for ticker in tickers:
        try:
            # Predict for each ticker
            signal = predictor.load_and_predict(ticker)
            print(f"Prediction for {ticker}: {signal}")
            
            # You would need the actual true labels for comparison to evaluate accuracy
            # Here we are just using a placeholder for the demonstration
            true_labels.append('Long')  # Replace with actual true label
            predicted_labels.append(signal)
        
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")

    # Evaluate model performance
    metrics = predictor.evaluate_model(true_labels, predicted_labels)
    print("Evaluation Metrics:", metrics)
