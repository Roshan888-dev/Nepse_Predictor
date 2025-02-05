import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LSTM, Conv1D, Dense, Dropout, LayerNormalization

class SMCTransformer(tf.keras.Model):
    def __init__(self, num_features, num_classes=3, 
                 num_heads=8, key_dim=64, lstm_units=256, 
                 conv_filters=128, dropout_rate=0.4):
        super().__init__()
        
        # Feature extraction
        self.conv1 = Conv1D(conv_filters, 5, activation='relu', padding='same')
        self.norm1 = LayerNormalization()
        
        # Temporal processing
        self.lstm = LSTM(lstm_units, return_sequences=True)
        self.dropout1 = Dropout(dropout_rate)
        
        # Attention mechanism
        self.attention = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=key_dim
        )
        self.norm2 = LayerNormalization()
        
        # Classifier
        self.classifier = tf.keras.Sequential([
            Dense(128, activation='gelu'),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.lstm(x)
        x = self.dropout1(x)
        
        # Self-attention
        attn_output = self.attention(x, x)
        x = self.norm2(x + attn_output)
        
        # Temporal pooling
        x = tf.reduce_mean(x, axis=1)
        
        return self.classifier(x)