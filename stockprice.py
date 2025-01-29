import yfinance as yf
import pandas as pd

# Define stock ticker symbol and data period
stock_ticker = "AAPL"  # Change to any stock symbol
data = yf.download(stock_ticker, start="2018-01-01", end="2023-01-01")

# Display data
print(data.head())
# Normalize the stock prices (for simplicity, using 'Close' price)
data['Normalized Close'] = data['Close'] / data['Close'].max()

# Handle missing values (if any)
data = data.fillna(method='ffill')

# Display preprocessed data
print(data[['Close', 'Normalized Close']].head())


import numpy as np
import pywt

# Fourier Transform
def apply_fourier_transform(data):
    return np.fft.fft(data)

# Wavelet Transform
def apply_wavelet_transform(data):
    coeffs = pywt.wavedec(data, 'db1', level=2)  # Daubechies wavelet
    return coeffs

# Apply Fourier Transform on 'Close' prices
data['Fourier'] = apply_fourier_transform(data['Normalized Close'])

# Apply Wavelet Transform on 'Close' prices
data['Wavelet'] = apply_wavelet_transform(data['Normalized Close'])

# Display results
print(data[['Close', 'Fourier', 'Wavelet']].head())
