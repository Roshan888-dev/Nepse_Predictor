import pywt
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, PowerTransformer
from config import DATA_CONFIG, PATHS

class UniversalDataProcessor:
    def __init__(self):
        self.scalers = {
            'price': RobustScaler(**DATA_CONFIG['scaler_params']['price']),
            'volume': PowerTransformer(**DATA_CONFIG['scaler_params']['volume'])
        }
        self.wavelet = DATA_CONFIG['wavelet']
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process raw data into clean, normalized features."""
        data = self._resample_and_clean(data)
        data = self._wavelet_denoise(data)
        data = self._add_technical_features(data)
        return self._scale_features(data)
    
    def _resample_and_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample data to 15-minute intervals and clean outliers."""
        data = data.resample('15T').last().ffill()
        vol_threshold = data['Volume'].rolling(100).quantile(0.25)
        return data[data['Volume'] > vol_threshold]
    
    def _wavelet_denoise(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply wavelet denoising to price columns."""
        for col in ['Open', 'High', 'Low', 'Close']:
            coeffs = pywt.wavedec(data[col], self.wavelet, mode='per')
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(data[col])))
            coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
            data[col] = pywt.waverec(coeffs, self.wavelet, mode='per')[:len(data)]
        return data
    
    def _add_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators like VWAP, ATR, and Fractal Energy."""
        data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
        data['ATR'] = data['High'].combine(data['Close'].shift(), lambda h, pc: h - pc).rolling(14).mean()
        data['FractalEnergy'] = data['Close'].rolling(50).apply(lambda x: np.sum(pywt.dwt(x, 'haar')[0]**2))
        return data
    
    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale price and volume features."""
        data[['Open', 'High', 'Low', 'Close']] = self.scalers['price'].fit_transform(data[['Open', 'High', 'Low', 'Close']])
        data['Volume'] = self.scalers['volume'].fit_transform(data[['Volume']])
        return data.dropna()

class MultiTimeFrameAnalyzer:
    def __init__(self):
        self.timeframes = DATA_CONFIG['timeframes']
        
    def analyze(self, data: pd.DataFrame) -> dict:
        """Analyze data across multiple timeframes."""
        results = {}
        for tf in self.timeframes:
            resampled = data.resample(tf).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            results[tf] = resampled
        return self._align_timeframes(results)
    
    def _align_timeframes(self, tf_data: dict) -> pd.DataFrame:
        """Align multi-timeframe data to a common index."""
        base_index = tf_data['D'].index
        aligned = {}
        for tf in self.timeframes:
            aligned[tf] = tf_data[tf].reindex(base_index, method='ffill')
        return pd.concat(aligned, axis=1)