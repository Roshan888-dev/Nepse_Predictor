import tensorflow as tf
from config import MODEL_CONFIG, DATA_CONFIG

class SMCTransformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self._build_network()
        
    def _build_network(self):
        """Build the multi-scale transformer architecture."""
        # Daily Stream
        self.daily_conv = tf.keras.layers.Conv1D(
            MODEL_CONFIG['conv_filters'], 5, activation='relu'
        )
        self.daily_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(MODEL_CONFIG['lstm_units'], return_sequences=False)  # Set return_sequences=False
        )
        
        # Weekly Stream
        self.weekly_conv = tf.keras.layers.Conv1D(
            MODEL_CONFIG['conv_filters'], 3, activation='relu'
        )
        self.weekly_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(MODEL_CONFIG['lstm_units'], return_sequences=False)  # Set return_sequences=False
        )
        
        # Monthly Stream
        self.monthly_conv = tf.keras.layers.Conv1D(
            MODEL_CONFIG['conv_filters'], 2, activation='relu'
        )
        self.monthly_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(MODEL_CONFIG['lstm_units'], return_sequences=False)  # Set return_sequences=False
        )
        
        # Attention Mechanism
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=MODEL_CONFIG['num_heads'],
            key_dim=MODEL_CONFIG['key_dim']
        )
        
        # Classifier
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(MODEL_CONFIG['dense_units'], activation='gelu'),
            tf.keras.layers.Dropout(MODEL_CONFIG['dropout_rate']),
            tf.keras.layers.Dense(3, activation='softmax')  # Long, Neutral, Exit
        ])
    
    def call(self, inputs):
        """Forward pass through the network."""
        # Process each timeframe
        daily_out = self.daily_lstm(self.daily_conv(inputs['daily']))
        weekly_out = self.weekly_lstm(self.weekly_conv(inputs['weekly']))
        monthly_out = self.monthly_lstm(self.monthly_conv(inputs['monthly']))
        
        # Stack outputs instead of concatenating
        fused = tf.stack([daily_out, weekly_out, monthly_out], axis=1)
        
        # Apply self-attention
        attended = self.attention(fused, fused)
        
        # Global average pooling
        pooled = tf.reduce_mean(attended, axis=1)
        
        # Classifier
        return self.classifier(pooled)

def create_multi_input_model() -> tf.keras.Model:
    """Create a multi-input model for daily, weekly, and monthly data."""
    daily_in = tf.keras.Input(
        shape=(DATA_CONFIG['sequence_lengths']['daily'], 7), name='daily'
    )
    weekly_in = tf.keras.Input(
        shape=(DATA_CONFIG['sequence_lengths']['weekly'], 7), name='weekly'
    )
    monthly_in = tf.keras.Input(
        shape=(DATA_CONFIG['sequence_lengths']['monthly'], 7), name='monthly'
    )
    
    # Create the transformer model
    transformer = SMCTransformer()
    
    # Define model outputs
    outputs = transformer({
        'daily': daily_in,
        'weekly': weekly_in,
        'monthly': monthly_in
    })
    
    return tf.keras.Model(
        inputs=[daily_in, weekly_in, monthly_in],
        outputs=outputs
    )