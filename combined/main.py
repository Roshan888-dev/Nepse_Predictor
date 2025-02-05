from data_preprocessing import SMCFeatureEngineer
from model_architecture import SMCTransformer
from train_evaluate import SMCTrainer
from yfinance import Ticker, download
import tensorflow as tf

if __name__ == "__main__":
    feature_engineer = SMCFeatureEngineer()
    model = SMCTransformer(num_features=6)
    trainer = SMCTrainer(model, feature_engineer)
    
    try:
        # Load model in .keras format
        model = tf.keras.models.load_model('smc_model.keras')
    except:
        print("Training new model...")
        trainer.train(['SPY', 'QQQ'], '2010-01-01', '2023-01-01')
        model.save('smc_model.keras')  # Save in Keras v3 format
    
    # Full market analysis
    results = trainer.analyze_market(universe='sp500')
    
    # Generate positions for top candidate
    top_pick = results[0]['ticker']
    data = download(top_pick, period='1y')
    features = feature_engineer.calculate_features(data)
    signals = trainer.generate_signals(data)
    positions = trainer.calculate_positions(data, signals)
    
    print(f"\nRecommended Position for {top_pick}:")
    print(positions.tail(10))