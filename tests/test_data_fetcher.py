import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.data_fetcher import generate_market_data
from src.data_manager.crypto_data_fetcher import CryptoDataFetcher
from config import CRYPTO_DATA_CONFIG

# Mock CryptoDataFetcher for testing generate_market_data
@pytest.fixture
def mock_crypto_data_fetcher():
    mock_fetcher = MagicMock(spec=CryptoDataFetcher)
    # Configure the mock to return a dummy DataFrame
    mock_fetcher.fetch_historical_data.return_value = {
        'data': pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
    }
    mock_fetcher.initialize_exchange.return_value = MagicMock() # Mock the exchange initialization
    return mock_fetcher

def test_generate_market_data_success(mock_crypto_data_fetcher):
    with patch('src.data_fetcher.CryptoDataFetcher', return_value=mock_crypto_data_fetcher):
        df = generate_market_data()
        
        assert not df.empty
        assert len(df) == 5
        assert 'open' in df.columns
        mock_crypto_data_fetcher.initialize_exchange.assert_called_once_with(CRYPTO_DATA_CONFIG['exchange'])
        mock_crypto_data_fetcher.fetch_historical_data.assert_called_once_with(
            mock_crypto_data_fetcher.initialize_exchange.return_value, # The mocked exchange object
            CRYPTO_DATA_CONFIG['symbol'],
            CRYPTO_DATA_CONFIG['timeframe']
        )

def test_generate_market_data_empty_return(mock_crypto_data_fetcher):
    mock_crypto_data_fetcher.fetch_historical_data.return_value = {'data': pd.DataFrame()}
    with patch('src.data_fetcher.CryptoDataFetcher', return_value=mock_crypto_data_fetcher):
        df = generate_market_data()
        assert df.empty

def test_generate_market_data_initialization_failure(mock_crypto_data_fetcher):
    mock_crypto_data_fetcher.initialize_exchange.return_value = None
    with patch('src.data_fetcher.CryptoDataFetcher', return_value=mock_crypto_data_fetcher):
        df = generate_market_data()
        assert df.empty