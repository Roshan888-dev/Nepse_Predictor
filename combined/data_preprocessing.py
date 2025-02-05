import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class SMCFeatureEngineer:
    def __init__(self, window=20):
        self.window = window
        self.scaler = StandardScaler()
        
    def calculate_features(self, df):
        df = df.copy()
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(self.window).std()
        
        df['market_structure'] = self._calculate_market_structure(df)
        df['order_block_strength'] = self._calculate_order_blocks(df)
        df['liquidity_gradient'] = self._calculate_liquidity_zones(df)
        df['imbalance_ratio'] = self._calculate_imbalance_ratio(df)
        df['wyckoff_phase'] = self._calculate_wyckoff_phases(df)
        
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
            
            if current_high > prev2_high and current_high > prev1_high:
                structure_series.iloc[i] = 1
            elif current_low < prev2_low and current_low < prev1_low:
                structure_series.iloc[i] = -1
            
            if (current_high < prev1_high and prev1_high > prev2_high and
                current_low > prev1_low and prev1_low < prev2_low):
                structure_series.iloc[i] = -2
            elif (current_low > prev1_low and prev1_low < prev2_low and
                current_high < prev1_high and prev1_high > prev2_high):
                structure_series.iloc[i] = 2
        
        return structure_series
    
    def _calculate_order_blocks(self, df):
        ob_strength = pd.Series(0.0, index=df.index)
        
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
            if (df['Volume'].iloc[i] > df['Volume'].iloc[i-1] and
                df['Close'].iloc[i] > vwap.iloc[i] and
                df['Low'].iloc[i] > df['Low'].iloc[i-2]):
                wyckoff.iloc[i] = 1
            elif (df['Volume'].iloc[i] > df['Volume'].iloc[i-1] and
                df['Close'].iloc[i] < vwap.iloc[i] and
                df['High'].iloc[i] < df['High'].iloc[i-2]):
                wyckoff.iloc[i] = -1
        return wyckoff