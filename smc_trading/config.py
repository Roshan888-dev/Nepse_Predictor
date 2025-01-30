from pathlib import Path

# Data Configuration
DATA_CONFIG = {
    'timeframes': ['D', 'W', 'M'],  # Daily, Weekly, Monthly
    'wavelet': 'sym15',  # Wavelet type for denoising
    'scaler_params': {
        'price': {'quantile_range': (5, 95)},  # Robust scaling for price data
        'volume': {}  # PowerTransformer for volume
    },
    'sequence_lengths': {
        'daily': 30,  # 30 days
        'weekly': 12,  # 12 weeks (~3 months)
        'monthly': 6  # 6 months
    }
}

# Model Architecture
MODEL_CONFIG = {
    'num_heads': 4,  # Multi-head attention heads
    'key_dim': 64,  # Attention key dimension
    'lstm_units': 128,  # LSTM hidden units
    'conv_filters': 64,  # Conv1D filters
    'dense_units': 128,  # Dense layer units
    'dropout_rate': 0.4,  # Dropout rate
    'learning_rate': 3e-4  # Learning rate for optimizer
}

# Trading Parameters
TRADING_CONFIG = {
    'risk_per_trade': 0.01,  # 1% risk per trade
    'stop_loss_multiplier': 1.5,  # ATR multiplier for stop loss
    'take_profit_multiplier': 3.0,  # ATR multiplier for take profit
    'max_position_duration': 21  # Max holding period in days
}

# Path Configuration
PATHS = {
    'models': Path('trained_models'),  # Directory for saving models
    'reports': Path('analysis_reports'),  # Directory for saving reports
    'data_cache': Path('.data_cache')  # Directory for caching data
}

# Ensure directories exist
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)