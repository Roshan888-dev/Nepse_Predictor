import pandas as pd
import numpy as np
from data_processor import UniversalDataProcessor

class SMCFeatureEngineer:
    def __init__(self):
        self.processor = UniversalDataProcessor()
        
    def calculate_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate SMC features from raw data."""
        data = self.processor.process(data)
        features = pd.DataFrame({
            'market_structure': self._calculate_market_structure(data),
            'order_block': self._calculate_order_blocks(data),
            'liquidity_gradient': self._calculate_liquidity_zones(data),
            'imbalance': self._calculate_imbalance_ratio(data),
            'wyckoff': self._calculate_wyckoff_phases(data),
            'fractal_energy': data['FractalEnergy'],
            'ATR': data['ATR']
        })
        return features.ffill().dropna().values
    
    def _calculate_market_structure(self, data: pd.DataFrame) -> pd.Series:
        """Calculate market structure features."""
        structure = pd.Series(0, index=data.index)
        highs, lows = data['High'], data['Low']
        
        for i in range(2, len(data) - 2):
            if highs.iloc[i] > max(highs.iloc[i-1], highs.iloc[i-2]):
                structure.iloc[i] = 1  # Bullish BOS
            elif lows.iloc[i] < min(lows.iloc[i-1], lows.iloc[i-2]):
                structure.iloc[i] = -1  # Bearish BOS
            
            if (highs.iloc[i] < highs.iloc[i-1] and lows.iloc[i] > lows.iloc[i-1] and
                highs.iloc[i-1] > highs.iloc[i-2] and lows.iloc[i-1] < lows.iloc[i-2]):
                structure.iloc[i] = -2  # Bearish CHOCH
            elif (lows.iloc[i] > lows.iloc[i-1] and highs.iloc[i] < highs.iloc[i-1] and
                  lows.iloc[i-1] < lows.iloc[i-2] and highs.iloc[i-1] > highs.iloc[i-2]):
                structure.iloc[i] = 2  # Bullish CHOCH
        
        return structure + 0.5 * (data['FractalEnergy'] > data['FractalEnergy'].quantile(0.75))
    
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