import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import pytz
import logging

# Suppress logging during tests for cleaner output
logging.basicConfig(level=logging.CRITICAL)

# Mock the external libraries
class MockCCXTExchange:
    def __init__(self, id='mock_exchange'):
        self.id = id
        self.symbols = ['BTC/USDT', 'ETH/USDT']

    def load_markets(self):
        pass

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        # Return dummy OHLCV data
        if symbol == 'BTC/USDT':
            return [[1678886400000, 100, 110, 90, 105, 1000]] * 5  # 5 data points
        return []

class MockYFinanceTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, period, interval):
        # Return dummy yfinance data
        if self.symbol == 'BTC-USD':
            dates = pd.to_datetime(['2023-01-01', '2023-01-02'])
            return pd.DataFrame({
                'Open': [100, 101],
                'High': [105, 106],
                'Low': [95, 96],
                'Close': [102, 103],
                'Volume': [1000, 1100]
            }, index=dates)
        return pd.DataFrame()

# Dummy config for testing
dummy_config = {
    'exchanges': ['mock_exchange'],
    'symbols': ['BTC/USDT'],
    'timeframes': ['1h'],
    'yfinance_symbols': ['BTC-USD'],
    'yfinance_intervals': ['1d'],
    'period': '1d',
    'start_date': datetime(2023, 1, 1, tzinfo=pytz.utc),
    'end_date': datetime(2023, 1, 2, tzinfo=pytz.utc),
    'max_retries': 1,
    'backoff_base': 1,
    'max_workers': 1,
    'refresh_interval': 1,
    'backup_dir': 'test_data_backup'
}

# Patch ccxt and yfinance globally for these tests
@patch('ccxt.binance', MockCCXTExchange)
@patch('yfinance.Ticker', MockYFinanceTicker)
@patch('redis.StrictRedis')
@patch('os.makedirs')
@patch('time.sleep')
def test_crypto_data_fetcher_initialization(mock_sleep, mock_makedirs, mock_redis):
    from src.data_manager.crypto_data_fetcher import CryptoDataFetcher
    fetcher = CryptoDataFetcher(dummy_config)
    assert fetcher.config == dummy_config

@patch('ccxt.binance', MockCCXTExchange)
@patch('yfinance.Ticker', MockYFinanceTicker)
@patch('redis.StrictRedis')
@patch('os.makedirs')
@patch('time.sleep')
def test_fetch_ohlcv_success(mock_sleep, mock_makedirs, mock_redis):
    from src.data_manager.crypto_data_fetcher import CryptoDataFetcher
    fetcher = CryptoDataFetcher(dummy_config)
    exchange = MockCCXTExchange()
    df = fetcher.fetch_ohlcv(exchange, 'BTC/USDT', '1h')
    assert not df.empty
    assert len(df) == 5
    assert 'open' in df.columns

@patch('ccxt.binance', MockCCXTExchange)
@patch('yfinance.Ticker', MockYFinanceTicker)
@patch('redis.StrictRedis')
@patch('os.makedirs')
@patch('time.sleep')
def test_fetch_yfinance_data_success(mock_sleep, mock_makedirs, mock_redis):
    from src.data_manager.crypto_data_fetcher import CryptoDataFetcher
    fetcher = CryptoDataFetcher(dummy_config)
    df = fetcher.fetch_yfinance_data('BTC-USD', '1d', '1d')
    assert not df.empty
    assert len(df) == 2
    assert 'Open' in df.columns

@patch('ccxt.binance', MockCCXTExchange)
@patch('yfinance.Ticker', MockYFinanceTicker)
@patch('redis.StrictRedis')
@patch('os.makedirs')
@patch('time.sleep')
def test_save_to_csv(mock_sleep, mock_makedirs, mock_redis):
    from src.data_manager.crypto_data_fetcher import CryptoDataFetcher
    fetcher = CryptoDataFetcher(dummy_config)
    dummy_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    path = "/tmp/test.csv"
    
    with patch.object(pd.DataFrame, 'to_csv') as mock_to_csv:
        fetcher.save_to_csv(dummy_df, path)
        mock_makedirs.assert_called_once_with(os.path.dirname(path), exist_ok=True)
        mock_to_csv.assert_called_once_with(path)

@patch('ccxt.binance', MockCCXTExchange)
@patch('yfinance.Ticker', MockYFinanceTicker)
@patch('redis.StrictRedis')
@patch('os.makedirs')
@patch('time.sleep')
def test_fetch_multiple_symbols(mock_sleep, mock_makedirs, mock_redis):
    from src.data_manager.crypto_data_fetcher import CryptoDataFetcher
    fetcher = CryptoDataFetcher(dummy_config)
    
    # Define dummy DataFrames separately
    dummy_ccxt_df = pd.DataFrame({'timestamp': [1], 'open': [1], 'high': [1], 'low': [1], 'close': [1], 'volume': [1]})
    dummy_yfinance_df = pd.DataFrame({'Open': [1]})

    # Mock the internal fetch methods to control their return values
    with (patch.object(fetcher, 'initialize_exchange', return_value=MockCCXTExchange()) as mock_initialize_exchange,
          patch.object(fetcher, 'fetch_historical_data', return_value={'source': 'ccxt', 'exchange': 'mock_exchange', 'symbol': 'BTC/USDT', 'timeframe': '1h', 'data': dummy_ccxt_df}) as mock_fetch_historical_data,
          patch.object(fetcher, '_fetch_yfinance_data_with_metadata', return_value={'source': 'yfinance', 'symbol': 'BTC-USD', 'interval': '1d', 'data': dummy_yfinance_df}) as mock_fetch_yfinance_data_with_metadata,
          patch.object(fetcher, 'save_to_csv') as mock_save_to_csv):

        fetcher.fetch_multiple_symbols()
        
        # Assert that fetch methods were called
        mock_fetch_historical_data.assert_called()
        mock_fetch_yfinance_data_with_metadata.assert_called()
        
        # Assert that save_to_csv was called for each fetched data
        assert mock_save_to_csv.call_count >= 2 # At least one for ccxt and one for yfinance